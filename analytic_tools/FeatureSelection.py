import argparse
import copy
import itertools
import logging
import os
import sys
import traceback

# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import numpy as np
import pandas as pd
import spektral
import tensorflow as tf
from tensorflow.keras import layers
from time import perf_counter
from datetime import datetime

from stgcn.STGCN import STGCN
from configuration.Configuration import Configuration
from execution.Evaluator import Evaluator
from stgcn.Dataset import Dataset
from configuration.Hyperparameter import Hyperparameter
from execution import GridSearch


def get_uncompiled_model(hyper, nbr_classes):
    input = tf.keras.Input(shape=(hyper.time_series_length, hyper.time_series_depth))
    x = input

    for f, s, d in zip(hyper.cnn_1D_filters, hyper.cnn_1D_filter_sizes, hyper.cnn_1D_dilation_rates):
        x = layers.Conv1D(filters=f, kernel_size=s, dilation_rate=d, padding=hyper.cnn_1D_padding)(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.ReLU()(x)

    x = layers.Conv1D(filters=hyper.time_series_depth, kernel_size=1, padding='causal')(x)
    x = layers.BatchNormalization(scale=False)(x)
    x = x + input
    x = layers.ReLU()(x)

    # Aggregate independently from the number of features such that the number of parameters stays the same
    x = layers.Permute(dims=[2, 1], name='pre_agg_perm')(x)
    x = spektral.layers.GlobalAttentionPool(name='agg_avg', channels=50)(x)

    out = layers.Dense(nbr_classes, name="out")(x)

    return tf.keras.Model(input, out)


def get_compiled_model(hyper: Hyperparameter, nbr_classes):
    # Create a simple Conv1D based neural network
    model = get_uncompiled_model(hyper, nbr_classes)

    if hyper.gradient_cap_enabled:
        optimizer = tf.optimizers.Adam(hyper.learning_rate, clipnorm=hyper.gradient_cap)
    else:
        optimizer = tf.optimizers.Adam(hyper.learning_rate)

    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.CategoricalAccuracy()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def train_and_test_simple_model(subset_config: Configuration, subset_dataset: Dataset, hyper: Hyperparameter):
    # Train a simple Conv1D based classifier
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=hyper.early_stopping_limit,
                                         restore_best_weights=True, verbose=1)
    ]

    model = get_compiled_model(hyper, subset_dataset.num_classes)
    model.fit(x=subset_dataset.get_train().get_x(), y=subset_dataset.get_train().get_y(),
              validation_data=(subset_dataset.get_train().get_x(), subset_dataset.get_train().get_y()),
              epochs=hyper.epochs, batch_size=hyper.batch_size, callbacks=[callbacks], shuffle=False, verbose=2)
    print()

    part_tested = subset_dataset.get_part(subset_config.part_tested)

    # Evaluate the trained model on the test dataset
    predictions_logits = model.predict(part_tested.get_x())
    predictions_softmax = tf.nn.softmax(predictions_logits).numpy()
    predictions_class_index = np.argmax(predictions_softmax, axis=1)
    predictions_class_label = subset_dataset.unique_labels_overall[predictions_class_index]

    evaluator = Evaluator(subset_dataset, part_tested=part_tested.get_type())
    evaluator.add_prediction_batch(np.arange(part_tested.get_num_instances()), predictions_class_label)

    return evaluator


def generate_feature_subsets(feature_selection_mode, feature_names, config):
    subsets = []

    # Randomly selects subset of features per test
    if feature_selection_mode == 'random':
        n_tests = 3
        subset_size_perc = 0.5

        subset_size = int(feature_names.shape[0] * subset_size_perc)
        rng = np.random.default_rng(seed=config.random_seed)

        for i in range(n_tests):
            subset = rng.choice(feature_names, size=subset_size, replace=False)
            subsets.append(subset)

        subset_descriptions = [f'random_subset_{i}' for i in range(n_tests)]

    # Leaves out features for which one of the substrings is in the feature name (same string per subset)
    elif feature_selection_mode == 'grouped':

        group_sub_strings = ['target_position']

        for substring in group_sub_strings:
            subset = [label for label in feature_names if substring not in label]
            subsets.append(subset)

        subset_descriptions = group_sub_strings

    # Leaves out features for which one of the substrings is in the feature name (one of all possible combinations)
    # E.g. for sub_strings = ['txt15', 'txt16']
    # subset 1: feature is included if it does not contain 'txt15'
    # subset 2: feature is included if it does not contain 'txt16'
    # subset 3: feature is included if it does not contain 'txt15' or 'txt16'
    elif feature_selection_mode == 'comb_grouped':

        sub_strings = []
        perms = []

        for l in range(1, len(sub_strings) + 1):
            perms.extend(list(itertools.combinations(sub_strings, l)))

        for substrings in perms:
            subset = [label for label in feature_names if not any(substring in label for substring in substrings)]
            subsets.append(subset)

        subset_descriptions = ['-'.join(perm) for perm in perms]

    elif feature_selection_mode == 'fixed_groups':
        groups, subset_descriptions = import_groups(config)

        for group in groups:
            subset = [label for label in feature_names if
                      not any(group_label == label for group_label in group)]
            subsets.append(subset)
    elif feature_selection_mode == 'fixed_groups_perm':
        groups, subset_descriptions = import_groups(config)
        perms, perms_desc = [], []

        r = range(2, 4)  # Combinations of 2 or 3
        # r = range(4, 6) # Combinations of 4 to 5
        # r = range(6, 8)  # Combinations of 6 to 7
        # r = range(1, len(groups) + 1)

        for l in r:
            perms.extend(list(itertools.combinations(groups, l)))
            perms_desc.extend(list(itertools.combinations(subset_descriptions, l)))

        groups, subset_descriptions = [], []
        for i, (perm, desc) in enumerate(zip(perms, perms_desc)):
            x = list(set(itertools.chain(*perm)))
            y = "-".join(sorted(list(itertools.chain(desc))))

            if y in subset_descriptions:
                raise ValueError()

            groups.append(x)
            subset_descriptions.append(y)

        groups, subset_descriptions = reduce_combinations(groups, subset_descriptions, 150,
                                                          random_seed=config.random_seed)

        for group in groups:
            subset = [label for label in feature_names if not any(group_label == label for group_label in group)]
            subsets.append(subset)

    else:
        raise ValueError()

    # Add baseline with all features
    subsets = [feature_names] + subsets
    subset_descriptions = ['baseline'] + subset_descriptions

    return subsets, subset_descriptions


def reduce_combinations(groups, subset_descriptions, limit, random_seed):
    if limit < len(subset_descriptions):
        print(f'Reducing number of combinations from {len(subset_descriptions)} to {limit}.\n')
    else:
        limit = len(subset_descriptions)

    rng = np.random.default_rng(random_seed)
    reduced_indices = rng.choice(a=np.arange(len(subset_descriptions)), size=limit, replace=False)

    groups = list(np.array(groups, dtype=object)[reduced_indices])
    subset_descriptions = list(np.array(subset_descriptions)[reduced_indices])

    return groups, subset_descriptions


def import_groups(config: Configuration):
    groups, descriptions = [], []

    # Imports subsets tested from a excel sheet named groups in the defined file.
    file = 'feature_selection.xlsx'
    feature_selection = pd.read_excel(config.get_additional_data_path(file), sheet_name='groups', engine='openpyxl')

    # Each column represents a subset. The column names is used as description.
    for col in feature_selection.columns:

        # Skip empty columns
        if col.startswith('Unnamed'):
            continue

        # Rows of column are the features removed for this group
        features = feature_selection[col].dropna().values
        features = sorted(list(set(features)))
        groups.append(features)
        descriptions.append(col)

    return groups, descriptions


def test_feature_subsets(config: Configuration, feature_subsets, subset_descriptions, all_feature_names, model_type):
    spacer = '**************************************************************' * 2
    fs_id = np.random.default_rng().integers(1000, 9999)
    nbr_configurations = len(feature_subsets)
    metrics = ['F1', 'F2', 'Recall', 'Precision']

    results_list = []
    result_dfs = np.empty((nbr_configurations,), dtype=object)

    hyper = Hyperparameter()
    hyper.load_from_file(config.hyper_file_folder + '_feature_selection.json')

    start = perf_counter()
    for index, feature_subset in enumerate(feature_subsets):

        STGCN.reset_state()
        print(spacer)
        print(f'Currently testing combination {index + 1}/{nbr_configurations} ...')
        print(spacer, '\n')

        # Create a deep copy of the gernal configuration object and alter the features that should be used.
        # and other necessary settings.
        # Primarily subsequent_feature_reduction must be true such that the dataset loaded using this object
        # is reduced to the subset of features currently tested.
        subset_config = copy.deepcopy(config)
        subset_config.subsequent_feature_reduction = True
        subset_config.create_a_plots = False
        subset_config.a_plot_display_labels = True
        subset_config.primary_features = feature_subset
        subset_config.features_used = sorted(
            list(set(subset_config.primary_features).union(subset_config.secondary_features)))

        subset_dataset = Dataset(subset_config)
        subset_dataset.load()
        hyper.set_time_series_properties(subset_dataset.time_series_length, subset_dataset.time_series_depth)

        # Standard training / test procedure, same as grid search.
        if model_type == 'simple':
            evaluator = train_and_test_simple_model(subset_config, subset_dataset, hyper)
        elif model_type == 'stgcn':
            stgcn = STGCN(subset_config, subset_dataset, training=True, hyper=hyper)
            stgcn.trainer.set_file_names('fs_{}_model_{}'.format(fs_id, index))
            selected_model_name = stgcn.train_model()
            stgcn.switch_to_testing()
            evaluator: Evaluator = stgcn.test_model(print_results=False, selected_model_name=selected_model_name)

        else:
            raise ValueError()

        if config.output_intermediate_results:
            evaluator.print_results(0)

        # Store the results of the current run
        r = evaluator.get_results()
        result_dfs[index] = r
        feature_subset_mask = [1 if feature in feature_subset else 0 for feature in all_feature_names]
        results_row = list(r.loc['weighted avg', metrics]) + feature_subset_mask
        results_list.append(results_row)

    end = perf_counter()

    cols = metrics + list(all_feature_names)
    results = pd.DataFrame(columns=cols, data=results_list)
    results.index.name = 'Comb'
    results['Description'] = subset_descriptions

    # higher = better is assumed
    results = results.sort_values(by=metrics, ascending=False)

    # For each feature calculate the average of the metrics for those run in which the feature was included
    feature_mean_lists = []
    for metric in metrics:
        means = []

        for feature in all_feature_names:
            tests_with_feature = results.loc[results[feature] == 1]
            scores = tests_with_feature[metric].values
            mean = np.mean(scores) if len(tests_with_feature) > 0 else np.NAN
            means.append(mean)
        feature_mean_lists.append(means)

    print(spacer)
    print('Feature Selection Results')
    print(spacer, '\n')

    print('\n\nBest combinations tested:\n')
    print(results.drop(columns=all_feature_names).head(150).round(5).to_string())
    if len(results) > 0:
        best_comb_index = results.index[0]
        print('\n\nFull result output for the best combination:\n')
        print(result_dfs[best_comb_index].round(5).to_string())
    print('\n\nExecution time: {}'.format(end - start))
    print(spacer, '\n')
    # means_df = pd.DataFrame(feature_mean_lists, columns=all_feature_names, index=metrics).T
    # print(means_df.to_string())
    # print(spacer, '\n')
    # print(means_df.sort_values(by=metrics, ascending=False).to_string())

    df_export = results.set_index('Description')
    for metric, means in zip(metrics, feature_mean_lists):
        df_export.loc[metric, all_feature_names] = means

    dt = datetime.now().strftime("%m-%d_%H-%M")
    df_export.head(150).round(5).to_excel(f'../logs/feature_selection_{dt}.xlsx')


def main():
    """
    Script used for running a feature selection using different feature selection techniques.
    Will train and test a stgcn or simple convolutional model on multiple subsets of features.
    """

    STGCN.reset_state()

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['random', 'grouped', 'comb_grouped', 'fixed_groups', 'fixed_groups_perm'],
                        required=True, help='')
    parser.add_argument('--model', choices=['simple', 'stgcn'], required=True, help='')
    args, _ = parser.parse_known_args()

    config = Configuration()
    GridSearch.modify_config(config)

    dataset = Dataset(config)
    dataset.load()

    # Get the subsets of features that are tested by training and testing models only using those.
    feature_subsets, subset_descriptions = generate_feature_subsets(args.method, dataset.feature_names, config)

    test_feature_subsets(config, feature_subsets, subset_descriptions, dataset.feature_names, args.model)


if __name__ == '__main__':
    # Try except around everything such that stack traces end up in the training log for debugging
    try:
        main()
    except:
        traceback.print_exc()
