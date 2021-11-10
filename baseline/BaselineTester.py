import itertools
import logging
import os
import sys
from multiprocessing import Pool
from time import perf_counter

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from sklearn.linear_model import RidgeClassifier

# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from baseline.SNN import SNN
from execution.Evaluator import Evaluator
from stgcn.Dataset import Dataset
from configuration.Configuration import Configuration
from configuration.Enums import BaselineAlgorithm as BA
from configuration.Enums import Representation as Rep


def print_config(config: Configuration):
    print('Representation used:', config.representation)
    print('Algorithm used:', config.baseline_algorithm)
    print('Dataset part tested:', config.part_tested)
    print('Inference of failures only:', config.test_failures_only)


def grid_to_lists(parameter_grid: dict):
    parameter_names, values = [], []
    for key, value in parameter_grid.items():
        parameter_names.append(key)
        values.append(value)

    grid_configurations = list(itertools.product(*values))

    return grid_configurations, parameter_names


def set_parameters(clf, parameter_names, config):
    for parameter, value in zip(parameter_names, config):

        # Ensure the key in hyperparameter defined in the json file exists as a hyperparameter
        if hasattr(clf, parameter):
            setattr(clf, parameter, value)
        else:
            raise ValueError('Unknown parameter: {}'.format(parameter))


def instance_chunk_calculation(config: Configuration, chunk_train, test_example, chunk_indices, chunk_id):
    """
    Calculates the distance / similarity of the current example test_example to all examples in the chunk of the
    training data sequentially.
    """
    chunk_results = np.zeros(chunk_indices.shape[0])

    for array_index, train_example in enumerate(chunk_train):

        if config.baseline_algorithm == BA.DTW:
            distance, _ = fastdtw(test_example, train_example, dist=2)
        else:
            raise ValueError('Algorithm not implemented.')

        chunk_results[array_index] = distance

    return chunk_id, chunk_results


def test_instance_based_algorithm(config: Configuration, dataset: Dataset, x_test, tested_indices):
    evaluator = Evaluator(dataset, part_tested=config.part_tested)
    y_pred = []
    x_train_labels = dataset.get_train().get_y_strings()

    start_time = perf_counter()

    for test_index in tested_indices:
        print(f'Testing example {test_index}/{len(tested_indices)}')

        num_train = dataset.get_train().get_num_instances()
        distances = np.zeros(num_train)

        # Programmatically ensure data < 4 gib is created if problem occurs
        # Split the training examples into chunks based on the number of parallelized calculations.
        chunks = np.array_split(np.arange(num_train), config.multiprocessing_limit * 2)
        async_results = []

        with Pool(processes=config.multiprocessing_limit) as pool:

            for chunk_id, chunk_indices in enumerate(chunks):
                args = (config, dataset.get_train().get_x()[chunk_indices], x_test[test_index], chunk_indices, chunk_id)
                result = pool.apply_async(instance_chunk_calculation, args)
                async_results.append(result)

            for chunk_id, chunk_indices in enumerate(chunks):
                chunk_id_retrieved, chunk_distances = async_results[chunk_id].get()
                assert chunk_id == chunk_id_retrieved, 'ID mismatch'
                distances[chunk_indices] = chunk_distances

        # Convert distances into predicted class label
        train_indices_ranked_by_dis = np.argsort(distances)
        nearest_neighbor_label = x_train_labels[train_indices_ranked_by_dis[0]]
        y_pred.append(nearest_neighbor_label)

    elapsed_time = perf_counter() - start_time
    evaluator.add_prediction_batch(tested_indices, y_pred)
    evaluator.print_results(elapsed_time)


def test_model_based_algorithm(config: Configuration, dataset: Dataset, x_test: np.ndarray, tested_indices: np.ndarray):
    x_train, y_train = dataset.get_train().get_x(), dataset.get_train().get_y_strings()

    if config.baseline_algorithm == BA.RIDGE_CLASSIFIER:
        params = get_ridge_params()
    elif config.baseline_algorithm == BA.SNN:
        params = get_snn_params()
    else:
        raise ValueError('Algorithm not implemented: {}'.format(config.baseline_algorithm))

    # Convert the grid search hyperparameter dictionary to individual configurations, that will be tested.
    grid_configurations, parameter_names = grid_to_lists(params)
    results = []

    print('Evaluating {} with {} configurations.\n'.format(config.baseline_algorithm.upper(), len(grid_configurations)))
    for index, parameters in enumerate(grid_configurations):

        if config.baseline_algorithm == BA.RIDGE_CLASSIFIER:
            classifier = RidgeClassifier()
        elif config.baseline_algorithm == BA.SNN:
            classifier = SNN(config, dataset)
        else:
            raise ValueError('Algorithm not implemented: {}'.format(config.baseline_algorithm))

        set_parameters(classifier, parameter_names, parameters)

        evaluator = Evaluator(dataset, part_tested=config.part_tested)

        print('Evaluating configuration {}/{}'.format(index + 1, len(grid_configurations)))
        start = perf_counter()

        classifier.fit(x_train, y_train)
        end_train = perf_counter()

        y_test_pred = classifier.predict(x_test[tested_indices])
        end_test = perf_counter()

        print('Evaluation finished.')
        print(f'\t Time for training: {round(end_train - start, 4)} Seconds')
        print(f'\t Time for testing: {round(end_test - end_train, 4)} Seconds')
        print(f'\t Time for testing per example: {round((end_test - end_train) / len(y_test_pred), 6)} Seconds')
        print(f'\t Time combined: {round(end_test - start, 4)} Seconds\n')

        evaluator.add_prediction_batch(tested_indices, y_test_pred)

        evaluator.calculate_results()
        result = evaluator.get_results()

        params: dict = classifier.get_params()
        result.loc['combined', 'Parameters'] = str(params)

        results.append(result)

    cols = ['F1', 'F2', 'Recall', 'Precision']
    combined_results = pd.DataFrame(columns=cols)

    for parameters, result in zip(grid_configurations, results):
        index = len(combined_results)
        combined_results.loc[index, cols] = result.loc['weighted avg', cols]
        combined_results.loc[index, 'Parameters'] = result.loc['combined', 'Parameters']

    combined_results = combined_results.sort_values(by=cols, ascending=False)

    print()
    print('Grid search results:')
    print(combined_results.to_string())
    print('\n')
    print('Best result in detail:')
    r: pd.DataFrame = results[combined_results.iloc[0].name].drop(columns=['Parameters'])
    print(r.to_string())

    return combined_results


def test_baseline():
    """
    This script can be used to execute DTW and other baseline methods for comparison with the STGCN approach.
    """

    config = Configuration()

    print_config(config)
    check_config(config)

    dataset = Dataset(config)
    dataset.load()

    x_test = dataset.get_part(config.part_tested).get_x()

    if config.test_failures_only:
        tested_indices = dataset.get_part(config.part_tested).get_failure_indices()
    else:
        tested_indices = np.arange(x_test.shape[0])

    if BA.is_instance_based(config.baseline_algorithm):
        test_instance_based_algorithm(config, dataset, x_test, tested_indices)
    else:
        return test_model_based_algorithm(config, dataset, x_test, tested_indices)


def multiple_snn_model_test():
    models = [
        'stgcn_gat_g_25011997',
        'stgcn_gcn_g_a_emb_v1_25011997',
        'stgcn_gcn_g_a_emb_v3_25011997',
        'stgcn_gcn_g_a_param_25011997',
        'stgcn_gat_g_19011940',
        'stgcn_gcn_g_a_emb_v1_19011940',
        'stgcn_gcn_g_a_emb_v3_19011940',
        'stgcn_gcn_g_a_param_19011940',
        'stgcn_gat_g_15071936',
        'stgcn_gcn_g_a_emb_v1_15071936',
        'stgcn_gcn_g_a_emb_v3_15071936',
        'stgcn_gcn_g_a_param_15071936',
        'stgcn_gat_g_12101964',
        'stgcn_gcn_g_a_emb_v1_12101964',
        'stgcn_gcn_g_a_emb_v3_12101964',
        'stgcn_gcn_g_a_param_12101964',
        'stgcn_gat_g_24051997',
        'stgcn_gcn_g_a_emb_v1_24051997',
        'stgcn_gcn_g_a_emb_v3_24051997',
        'stgcn_gcn_g_a_param_24051997',
    ]

    overall_results = pd.DataFrame(columns=['Seed', 'File name', 'Comb', 'F1', 'F2', 'Recall', 'Precision'])

    for i, desc in enumerate(models):
        parts = desc.split('_')
        seed = int(parts[-1])
        model_dir = desc
        model_name = '_'.join(parts[0:-1])

        print('#####################################################')
        print(i, seed, model_name, model_dir)
        print('#####################################################')

        config = Configuration()
        config.filename_model_to_use = model_dir
        config.directory_model_to_use = config.models_folder + config.filename_model_to_use + '/'

        print_config(config)
        check_config(config)

        dataset = Dataset(config)
        dataset.load()

        x_test = dataset.get_part(config.part_tested).get_x()

        if config.test_failures_only:
            tested_indices = dataset.get_part(config.part_tested).get_failure_indices()
        else:
            tested_indices = np.arange(x_test.shape[0])

        single_result = test_model_based_algorithm(config, dataset, x_test, tested_indices)

        row = [seed, model_dir, model_name] + list(single_result.loc[0, ['F1', 'F2', 'Recall', 'Precision']].values)
        overall_results.loc[i] = row
        print()

    overall_results = overall_results.sort_values(by='F1', ascending=False)
    print('\n#####################################################')
    print('Final results:')
    print(overall_results.to_string())
    print('#####################################################')

    overall_results.round(5).to_excel(f"../logs/snn_test_results.xlsx")


def check_config(config: Configuration):
    """
    Checks whether the current configuration of selected algorithm and data representation can be used.
    :param config: A configuration object.
    """

    config_rep = config.representation
    config_algo = config.baseline_algorithm

    valid_configs = [
        Rep.contains_feature_vectors(config_rep) and config_algo in [BA.RIDGE_CLASSIFIER, BA.EUCLIDEAN_DISTANCE],
        not Rep.contains_feature_vectors(config_rep) and config_algo in [BA.DTW, BA.SNN]
    ]

    if not any(valid_configs):
        raise ValueError('Configured representation does not match the configured baseline algorithm: ',
                         config.representation, config.baseline_algorithm)


def get_ridge_params():
    """
    :return: A dictionary of hyperparameters required by the ridge classifier, each hp key has a list of values that
        should be tested via grid search.
    """

    # Parameters tested for the ridge configuration.
    # ridge_params = {
    #     "alpha": [0.001, 0.01, 0.1, 1, 5],
    #     "normalize": [True, False],
    #     "class_weight": [None, 'balanced'],
    #     "random_state": [25011997]
    # }

    # Selected by grid search of params above on test_val
    ridge_params = {
        "alpha": [0.1],
        "normalize": [False],
        "class_weight": ['balanced'],
        "random_state": [25011997, 19011940, 15071936, 12101964, 24051997]
    }

    return ridge_params


def get_snn_params():
    """
    :return: A dictionary of hyperparameters required by the SNN classifier, each hp key has a list of values that
        should be tested via grid search.
    """

    snn_params = {
        "k": [1],
        "sim_norm": [1],
        "drop_dense": [False],
    }

    return snn_params


if __name__ == '__main__':
    try:
        test_baseline()
        # multiple_snn_model_test()
    except KeyboardInterrupt:
        sys.exit(0)
