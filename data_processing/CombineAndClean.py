import argparse
import datetime as dt
import gc
import json
import os
import re
import sys
from datetime import datetime
from multiprocessing import Pool
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration


def combine_topics(dataset_path, storage_path, list_of_topics, list_name, start_time, end_time):
    result_df = pd.DataFrame()

    for i, topic in enumerate(list_of_topics):
        print('Importing topic {} of list {} ({}/{})'.format(topic, list_name, i, len(list_of_topics)))

        df: pd.DataFrame = pd.read_pickle(dataset_path + topic + '_pkl')
        result_df = result_df.join(df, how='outer')

    result_df = result_df[start_time: end_time]
    result_df.to_pickle(storage_path + list_name + '.pkl')

    print('Finished topic list {}'.format(list_name))
    return None


def fuse_topics():
    config = Configuration()

    for dataset_entry in config.stored_datasets:
        dataset_name: str = dataset_entry[0]
        dataset_name = dataset_name if dataset_name.endswith('/') else dataset_name + '/'

        dataset_path = config.kafka_imported_topics_path + dataset_name
        storage_path = config.unprocessed_data_path + dataset_name

        if not os.path.exists(storage_path):
            os.mkdir(storage_path)

        spacer = '---------------------------------------------------------------------------------------'
        print(spacer)
        print('Start creation of four .pkl files for dataset: {}'.format(dataset_name))
        print(spacer)

        start_time = datetime.strptime(dataset_entry[1], config.dt_format) - dt.timedelta(0, 1)
        end_time = datetime.strptime(dataset_entry[2], config.dt_format) + dt.timedelta(0, 1)

        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        # For future reference:
        # https://stackoverflow.com/questions/48932757/python-multiprocessing-return-values-from-3-different-functions
        with Pool(processes=config.multiprocessing_limit) as pool:

            r_txt = pool.apply_async(combine_topics, (dataset_path, storage_path,
                                                      list(config.kafka_txt_topics.keys()),
                                                      'txt_topics', start_time, end_time))

            r_sensors = pool.apply_async(combine_topics, (dataset_path, storage_path,
                                                          list(config.kafka_sensor_topics.keys()), 'sensor_topics',
                                                          start_time, end_time))

            r_webserver = pool.apply_async(combine_topics, (dataset_path, storage_path,
                                                            list(config.kafka_webserver_topics.keys()),
                                                            'webserver_topics', start_time, end_time))

            r_failure_simulation = pool.apply_async(combine_topics, (dataset_path, storage_path,
                                                                     list(config.kafka_failure_sim_topics.keys()),
                                                                     'failure_sim_topics', start_time, end_time))

            # Get return values = wait until finished.
            _, _, _, _ = r_txt.get(), r_sensors.get(), r_webserver.get(), r_failure_simulation.get()

        gc.collect()
        print('\n' + spacer + '\n')


def import_df(path, reduce, categorical_nan_replacement, features_used=None, ):
    df_name = path.split('/')[-1]
    df: pd.DataFrame = pd.read_pickle(path)

    # print('{} before: {} x {}'.format(df_name, len(df), len(df.columns)))

    if reduce:
        # Remove columns that should not be in the final dataset
        df = df[df.columns.intersection(features_used)]

    # Here None/'' values have the meaning 'sensor detected no signal'
    # np.NaN values have the meaning 'no sensor value for this timestamp'
    # # Because pandas can differentiate between NaN and None we can't use a simple replace
    # Not in seconds must be included such that current_task_elapsed_seconds are not set to NaN
    cols = [col for col in df.columns if
            any(key_word in col for key_word in categorical_nan_replacement) and not 'seconds' in col]
    for col in cols:
        # First use a type check to only set NaNs to a placeholder string (NaNs are of type float, Nones are not)
        df[col] = df[col].apply(lambda v: 'is_nan' if isinstance(v, float) else v)
        # Then set all other missing values to a new categorical value
        df[col] = df[col].replace(to_replace=['', 'no_reading', None], value='.no_reading')
        # Replace the placeholder string actual NaNs again
        df[col] = df[col].replace(to_replace=['is_nan'], value=np.NaN)

    # Sort columns alphabetically for easier interpretation for feature selection.
    # df = df.reindex(sorted(df.columns), axis=1)
    # #df = df.iloc[:, 220:]
    # df = df

    # Better way to handle this?
    # Replace all other invalid values with np.NaN such that the following drop work correctly.
    df = df.replace(r'^\s*$', np.NaN, regex=True)

    # First, remove all columns that contain any data at all, this should never be the case, will be checked later.
    df = df.dropna(axis='columns', how='all')

    # Remove rows for which all remaining columns are empty
    df = df.dropna(axis='index', how='all')

    # print('{} after: {} x {}'.format(df_name, len(df), len(df.columns)))

    if df.empty:
        print('WARNING for {}: {} is empty, no features used or all fully empty'.format(path.split('/')[-2], df_name))

    return df


def first_cc_step(dataset_name, resample_origin_time, config: Configuration):
    print('Started first cc step for dataset {}\n'.format(dataset_name))

    # If no split of features into a primary/secondary set is configured, only the primary ones need to be processed.
    features_used = config.features_used if config.create_secondary_feature_set else config.primary_features
    # Should be true except for debugging in order to increase performance.
    reduce = True
    cnr = config.categorical_nan_replacement
    dataset_path = config.unprocessed_data_path + dataset_name
    file_name = config.combined_and_cleaned_data_path + dataset_name + '.pkl'
    resample_freq = str(config.resample_frequency) + 'ms'
    start = perf_counter()

    df_txt = import_df(dataset_path + '/txt_topics.pkl', reduce, cnr, features_used)
    df_sensors = import_df(dataset_path + '/sensor_topics.pkl', reduce, cnr, features_used)

    # Combine into single large dataframe.
    df = pd.DataFrame()
    df = df.join(df_txt, how='outer')
    df = df.join(df_sensors, how='outer')

    try:
        df_webserver = import_df(dataset_path + '/webserver_topics.pkl', reduce, cnr, features_used)
        df = df.join(df_webserver, how='outer')
    except FileNotFoundError:
        pass

    try:
        df_failure_simulation = import_df(dataset_path + '/failure_sim_topics.pkl', reduce, cnr, features_used)
        df = df.join(df_failure_simulation, how='outer')
    except FileNotFoundError:
        pass

    del df_txt, df_sensors, df_failure_simulation, df_webserver
    gc.collect()

    # Check for columns that should be included but can not be found in the data loaded.
    x = [col for col in features_used if col not in df.columns]
    if len(x) > 0:
        print(f'\nWARNING for {dataset_name}: Columns that are defined in features_used but not recorded:')
        print(*x, sep='\n')

    # Check if a type is assigned in config.json for all columns.
    specified_cols = set(config.categorical_features + config.integer_features + config.float_features)
    x = [col for col in df.columns if col not in specified_cols]
    if len(x) > 0:
        print('\nWARNING: Contains columns that are not assigned to a column type list '
              '(which will lead to no nan filling and ultimately dropping the column):')
        print(*x, sep='\n')
        print()
        # raise ValueError()

    # Get lists features grouped by type which are also contained in the dataframe.
    used_and_included = set(features_used).intersection(df.columns)
    cat_included = list(set(config.categorical_features).intersection(used_and_included))
    int_included = list(set(config.integer_features).intersection(used_and_included))
    float_included = list(set(config.float_features).intersection(used_and_included))

    # Please don't try to improve the part until and including the resampling.
    # The order of operations and selected methods for filling values are mandatory.
    # Total hours wasted: 12

    # Replace NA values by forward filling
    df[cat_included + int_included] = df[cat_included + int_included].fillna(method='ffill')
    gc.collect()

    # Ensure all float valued streams are actually in a numeric column type.
    for col in float_included:
        df[col] = pd.to_numeric(df[col])
    gc.collect()

    # Interpolate NA values for float valued streams via a linear interpolation.
    df[float_included] = df[float_included].apply(pd.Series.interpolate, method='linear',
                                                  limit_direction='forward', axis=0)
    gc.collect()

    # Important: Forward filling also real values instead of lin. inter. works better here.
    df = df.resample(resample_freq, origin=resample_origin_time).pad()
    gc.collect()

    cols_removed, nbr_nans = [], []
    nbr_records = len(df)
    nans: pd.Series = df.isna().sum(axis=0)
    for attribute, nbr_of_nans in nans.items():
        if nbr_of_nans > 0.04 * nbr_records:
            cols_removed.append(attribute)
            nbr_nans.append(nbr_of_nans)

    if len(cols_removed) > 0:
        print('WARNING for {}:'.format(dataset_name))
        print('Some columns contain many NaNs. Most likely due to bad recording quality.')
        print('These were removed (attribute, rows with NaN):')
        print(*zip(cols_removed, nbr_nans), sep='\n')
        print()

        df = df.drop(columns=cols_removed)

    # Remove rows for which all remaining columns are empty.
    df = df.dropna(axis='index', how='any')
    gc.collect()

    # Sort columns alphabetically.
    df = df.reindex(sorted(df.columns), axis=1)

    # Final check if all feature that should in the final dataset are actually contained in the preprocessed data.
    not_present = [feature for feature in features_used if feature not in df.columns]
    if len(not_present) > 0:
        print('WARNING for {}:'.format(dataset_name))
        print('Not all features defined in features_used are present in the processed dataframe.')
        print('Most likely cause: Not recorded, therefore removed due to missing values.\n')
        print(*not_present, sep='\n')
        print()

    unique_values_in_cols = {}
    for col in df.columns:
        if col in cat_included:
            unique_values_in_cols[col] = df[col].unique()

    df.to_pickle(file_name)

    end = perf_counter()
    print(f'Finished first cc step for {dataset_name} in {round(end - start, 2)} seconds. Shape: {df.shape}\n')

    del df
    gc.collect()

    return unique_values_in_cols


def rename_columns(current_col_name, new_labels_mapping, predefined_mapping):
    new_col_name = current_col_name

    if current_col_name in new_labels_mapping.keys():
        new_col_name = new_labels_mapping[current_col_name]

    if new_col_name in predefined_mapping.keys():
        new_col_name = predefined_mapping[new_col_name]

    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    new_col_name = pattern.sub('_', new_col_name).lower()

    return new_col_name


def second_cc_step(config: Configuration, dataset_name, unique_values_combined, new_labels_mapping):
    print('Started second cc step for dataset {}\n'.format(dataset_name))
    start = perf_counter()

    cat_cols = list(unique_values_combined.keys())
    file_name = config.combined_and_cleaned_data_path + dataset_name + '.pkl'
    df: pd.DataFrame = pd.read_pickle(file_name)
    values = df[cat_cols].values

    unique_values_per_col = [unique_values_combined[col] for col in cat_cols]

    # Drop='first' not only reduces the number of features but is also recommended for neural networks.
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    drop_setting = 'first' if config.drop_first_one_hot_column else 'if_binary'
    one_hot_encoder = OneHotEncoder(categories=unique_values_per_col, drop=drop_setting, sparse=False)
    transformed = one_hot_encoder.fit_transform(values)
    new_col_names = one_hot_encoder.get_feature_names(cat_cols)

    # Order is important because transformed column names may match the old ones.
    df = df.drop(columns=cat_cols)
    df[new_col_names] = transformed

    # Correct the column names (all, not only the ones created by the one hot encoding).
    # Includes static rules like lower case conversion but also full renaming based on configuration.
    df = df.rename(columns=lambda c: rename_columns(c, new_labels_mapping, config.label_renaming_mapping))

    # Remove configured "new" columns that are the result of the one hot encoding.
    # Errors can be ignored, can only occur during debugging when using a subset of datasets.
    df = df.drop(columns=config.one_hot_features_to_remove, errors='ignore')

    # Sort columns alphabetically. Among other things, to match the adjacency matrices used.
    df = df.reindex(sorted(df.columns), axis=1)

    # Overwrite the file stored in the first step.
    df.to_pickle(file_name)

    end = perf_counter()
    print(f'Finished second step for {dataset_name} in {round(end - start, 2)} seconds. Shape: {df.shape}\n')

    cols = list(df.columns)
    del df
    gc.collect()

    return cols


def aggregate_unique_values(config: Configuration, unique_value_dicts):
    """
    Combines list of dicts to single dict with list values with same key are combined.
    """

    unique_values_in_all = {}
    for dict in unique_value_dicts:
        for key, value in dict.items():

            if key in unique_values_in_all.keys():
                unique_values_in_all[key].append(value)
            else:
                unique_values_in_all[key] = [value]

    for col in unique_values_in_all.keys():
        unique_values_in_all[col] = list(np.unique(np.concatenate(unique_values_in_all[col], axis=0)))

    # Regex for converting to snake case.
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    new_labels_mapping = {}
    exported_mappings = {}

    for col, unique_values in unique_values_in_all.items():
        for index, value in enumerate(unique_values):

            if not isinstance(value, str):
                value = str(value)

            # Create the label that will be created by the one hot encoder as key for the mapping.
            one_hot_label = col + '_' + str(value)

            # For boolean feature one hot encoder generates a unnecessarily complex labelling that should be removed.
            suffixes_to_remove = ['_1.0', '_True', '_1']
            for suffix in suffixes_to_remove:

                # Only remove 1 if boolean attribute is the reason,
                # not an actual categorical value e.g. number of stones in warehouse.
                if suffix in one_hot_label and len(unique_values) == 2:
                    new_label = one_hot_label.split(suffix)[0]
                    break
            else:
                # Categorical feature if here, create a simpler label.
                new_label = col + '_0' + str(index) if index < 10 else col + '_' + str(index)

            # Converts to snake case
            new_label = pattern.sub('_', new_label).lower()
            new_labels_mapping[one_hot_label] = new_label
            exported_mappings[new_label] = (col, value)

    # print(json.dumps(unique_values_in_all, sort_keys=False, indent=4))
    # print(json.dumps(new_labels_mapping, sort_keys=False, indent=4))
    # print(json.dumps(exported_mappings, sort_keys=False, indent=4))

    with open(config.get_additional_data_path('cat_feature_mapping.json'), 'w') as outfile:
        json.dump(exported_mappings, outfile, sort_keys=False, indent=2)

    return unique_values_in_all, new_labels_mapping


def combine_and_clean_dfs():
    config = Configuration()

    parser = argparse.ArgumentParser()
    parser.add_argument("--fuse_topics", default=False, action="store_true")
    parser.add_argument("--force_limit", default=False, action="store_true")
    args, _ = parser.parse_known_args()

    if args.fuse_topics:
        fuse_topics()

    if not args.force_limit and config.multiprocessing_limit > 5:
        raise ValueError('multiprocessing_limit probably set too high, value of 5 or lower recommended.'
                         ' Use --force_limit to ignore this error.')

    datasets, origin_times, async_results, unique_value_dicts = [], [], [], []

    for dataset_entry in config.stored_datasets:
        dataset_name: str = dataset_entry[0]
        dataset_resampling_origin_time = datetime.strptime(dataset_entry[1], '%Y-%m-%d %H:%M:%S.%f')
        dataset_name = dataset_name[0:-1] if dataset_name.endswith('/') else dataset_name
        datasets.append(dataset_name)
        origin_times.append(dataset_resampling_origin_time)

    # Execution for single dataset, for development purposes.
    # tested = 8
    # x = first_cc_step(datasets[tested], origin_times[tested], config)
    # x, new_labels_mapping = aggregate_unique_values(config, [x])
    # x = second_cc_step(config, datasets[tested], x, new_labels_mapping)
    # print(*x, sep='\n')
    # print('WARNING: ONLY COMBINED SINGLE ONE FOR DEBUGGING')
    # sys.exit(0)

    ####################################################################################################################
    # First part: Combine 4 Topic DFs into a single one, fill missing values, reduce to relevant features.
    ####################################################################################################################

    with Pool(processes=config.multiprocessing_limit) as pool:

        for dataset, origin_time in zip(datasets, origin_times):
            result = pool.apply_async(first_cc_step, (dataset, origin_time, config))
            async_results.append(result)

        for result in async_results:
            unique_value_dicts.append(result.get())

    print('\nFirst cc step finished for all all datasets.\n')

    async_results, df_cols = [], []
    unique_values_combined, new_labels_mapping = aggregate_unique_values(config, unique_value_dicts)

    ####################################################################################################################
    # Second part: Transform categorical features into a one hot encoding, sort columns.
    ####################################################################################################################

    with Pool(processes=config.multiprocessing_limit) as pool:

        for dataset in datasets:
            result = pool.apply_async(second_cc_step, (config, dataset, unique_values_combined, new_labels_mapping))
            async_results.append(result)

        for result in async_results:
            df_cols.append(result.get())

    if len(df_cols) > 1:
        df_0_cols = df_cols[0]

        for index, df_cols in enumerate(df_cols):
            if not np.array_equal(df_0_cols, df_cols):
                print('\nWARNING: Features in dataframes do not match:')
                print('Features in 0.:')
                print(*df_0_cols, sep='\n')
                print()
                print(f'Features in {index}.:')
                print(*df_cols, sep='\n')
                print()
                break
        else:
            if not os.path.exists(config.get_training_data_path()):
                os.makedirs(config.get_training_data_path())

            feature_names = np.array(df_0_cols)
            np.save(config.get_training_data_path() + 'feature_names.npy', feature_names)
            print('Features in stored dataframes:')
            print(f'(Only primary ones: {config.create_secondary_feature_set})\n')
            print(*feature_names, sep='\n')
            print()


if __name__ == '__main__':
    combine_and_clean_dfs()
