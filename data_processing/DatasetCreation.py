import argparse
import contextlib
import glob
import os
import sys
from multiprocessing import Pool
from time import sleep

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from data_processing.DSCleaner import DSCleaner, Normaliser
from configuration.Configuration import Configuration
from configuration.Enums import SplitMode, DatasetPart, Representation
from configuration.Enums import Representation as Rep


def no_mix_split(config, labels, runs, aux_index, split_size=None, runs_required=2):
    # Convert failure time strings to integer encoding.
    enc = OrdinalEncoder()
    enc.fit(runs.reshape(-1, 1))
    int_runs = np.squeeze(enc.transform(runs.reshape(-1, 1)).astype(int))

    # Used to iterate over each label and run.
    unique_runs = np.unique(int_runs)
    unique_labels = np.unique(labels)

    # Overall result lists containing the example.
    primary_indices, secondary_indices = [], []

    is_train_test_split = split_size is None
    split_size = config.test_split_size if is_train_test_split else split_size

    for label in unique_labels:
        if label in config.unused_labels:
            print(f'WARNING: Skipping {label} because it is contained in config.unused_labels\n')
            continue

        # Calculate the overall number of examples with this label and based on this and the configured test split size
        # how many of those should end up in test.
        indices_current_label = np.squeeze(np.argwhere(labels == label))

        # If only single element in the np.argwhere returned array squeeze will reduce it to a integer.
        try:
            nbr_examples_current_label = len(indices_current_label)
        except TypeError:
            nbr_examples_current_label = 1

        if is_train_test_split and config.use_separate_nf_split_size and label == 'no_failure':
            target_nbr_examples_secondary = int(np.ceil(nbr_examples_current_label * config.nf_test_split_size))
        else:
            target_nbr_examples_secondary = int(np.ceil(nbr_examples_current_label * split_size))

        indices_split_by_run = []

        for int_run in unique_runs:
            # Get the indices of examples that have the current label and were recorded during the current run.
            matching_example_indices = np.logical_and(labels == label, int_runs == int_run).nonzero()[0]
            # print(f'Example indices with label {label} and run {int_run}: {len(matching_example_indices)}')

            # This label did not occur in the current run.
            if len(matching_example_indices) != 0:
                indices_split_by_run.append(matching_example_indices)

        if len(indices_split_by_run) < runs_required:
            print(f'WARNING: Less than {runs_required} runs for label {label} '
                  f'-> Can not be part of a "ensure no mix" dataset.\n')
            continue

        indices_split_by_run: list = sorted(indices_split_by_run, key=len)

        nbr_examples_assigned_secondary = 0

        print(f'Current label: {label}\n')
        for index, run_group in enumerate(indices_split_by_run):
            examples_in_group = len(run_group)

            interval_id = runs[run_group[0]]

            # Assign runs to train or test based on the following criteria:
            # - Assign every second one to train. This ensures we also have short runs in train such that we avoid a possible bias
            # - Assign to test if new number in test does not exceed the configured

            # Newly added examples should lead to the number of examples in test exceeding the configured split size.
            # This should be allowed if otherwise no runs are assigned to test though.
            if nbr_examples_assigned_secondary > 0:
                limit_reached = nbr_examples_assigned_secondary + examples_in_group > target_nbr_examples_secondary
            else:
                limit_reached = False

            if index % 2 == 0 and not limit_reached:
                secondary_indices.extend(run_group)
                nbr_examples_assigned_secondary += examples_in_group
                print('\tGroup {} from interval {} assigned to secondary set:'.format(index, interval_id))
            else:
                primary_indices.extend(run_group)
                print('\tGroup {} from interval {} assigned to primary set:'.format(index, interval_id))
            print('\t\t# Examples in group:', examples_in_group)
            print('\t\t# Examples assigned to secondary now: {}/{}'.format(nbr_examples_assigned_secondary,
                                                                           target_nbr_examples_secondary))
            # print('\t', aux_index[run_group])
            print()

        print()
        print('\t# Examples with this label:', nbr_examples_current_label)
        print('\t# Examples in secondary target:', target_nbr_examples_secondary)
        print('\t# Examples in secondary:', nbr_examples_assigned_secondary)
        print('\t# Examples in primary:', nbr_examples_current_label - nbr_examples_assigned_secondary)
        print()

    primary_indices, secondary_indices = aux_index[primary_indices], aux_index[secondary_indices]
    return primary_indices, secondary_indices


def determine_val_indices(config: Configuration, aux_df: pd.DataFrame, main_indices, val_split_size):
    if val_split_size <= 0.0:
        return main_indices, []

    aux_df_part = aux_df.loc[main_indices]
    labels = aux_df_part['label'].values
    interval_ids = aux_df_part['r2f_id'].values
    aux_df_part_index = aux_df_part.index.values

    if config.split_mode_val == SplitMode.ENSURE_NO_MIX:
        print('\n#### VAL Split Start ####\n')
        main_indices_new, val_indices = no_mix_split(config, labels, interval_ids, aux_df_part_index,
                                                     val_split_size, runs_required=2)
        print('\n#### VAL Split End ####\n')
    elif config.split_mode_val == SplitMode.SEEDED_SK_LEARN:
        main_indices_new, val_indices = train_test_split(main_indices, test_size=val_split_size,
                                                         random_state=config.random_seed, stratify=labels)
    else:
        raise NotImplementedError('SplitMode not implemented.')

    return main_indices_new, val_indices


def determine_train_test_indices(config: Configuration, aux_df: pd.DataFrame):
    # Note that the r2f_id, not the interval id is used.
    labels = aux_df['label'].values
    interval_ids = aux_df['r2f_id'].values
    aux_index = aux_df.index.values

    print('\n#### Train/Test Split Start ####\n')
    if config.split_mode == SplitMode.ENSURE_NO_MIX:

        if not config.split_mode_val == SplitMode.ENSURE_NO_MIX:
            runs_required = 2
        else:
            if config.train_val_split_size > 0 and config.test_val_split_size > 0:
                runs_required = 4
            elif config.train_val_split_size > 0 or config.test_val_split_size > 0:
                runs_required = 3
            else:
                runs_required = 2

        train_indices, test_indices = no_mix_split(config, labels, interval_ids, aux_index, runs_required=runs_required)

    elif config.split_mode == SplitMode.SEEDED_SK_LEARN:
        train_indices, test_indices = train_test_split([i for i in range(len(labels))],
                                                       test_size=config.test_split_size,
                                                       random_state=config.random_seed,
                                                       stratify=labels)
    elif config.split_mode == SplitMode.ANOMALY_DETECTION:

        # This means all failure examples are in test.
        # Only no_failure examples will be split based on configured percentage.
        print('\nExecute train/test split in anomalie detection mode. '
              'WARNING: IMPLEMENTATION NOT TESTED YET. Use at your own risk.')

        # Split examples into normal and failure cases.
        failure_indices = np.argwhere(labels != 'no_failure').flatten()
        no_failure_indices = np.argwhere(labels == 'no_failure').flatten()

        # Execute recording instance based splitting only for no_failures.
        # For which the input arrays are first of all reduced to those examples.
        nf_labels = labels[no_failure_indices]
        nf_failure_times = interval_ids[no_failure_indices]
        nf_aux_index = aux_df[no_failure_indices].index.values

        nf_train_indices_in_reduced, nf_test_indices_in_reduced = no_mix_split(config, nf_labels,
                                                                               nf_failure_times, nf_aux_index)

        # Trace back the indices of the reduced arrays to the indices of the complete arrays.
        nf_train_indices = no_failure_indices[nf_train_indices_in_reduced]
        nf_test_indices = no_failure_indices[nf_test_indices_in_reduced]

        # Combine indices to full lists.
        # Train part only consists of the  train part of the no failure split,
        # whereas the test part consists of the test part of the no failure split as well as failure examples.
        train_indices = list(nf_train_indices)
        test_indices = list(failure_indices) + list(nf_test_indices)


    else:
        raise ValueError()

    print('\n#### Train/Test Split End ####\n')
    return train_indices, test_indices


def split_by_label(full_r2f: pd.DataFrame, r2f_info: pd.DataFrame):
    label_interval_dfs = {}

    for interval in range(len(r2f_info)):
        single_interval_info = r2f_info.loc[interval]
        start_timestamp = single_interval_info['start']
        end_timestamp = single_interval_info['end']
        interval_id = single_interval_info['interval_id']

        # Basic checks for correct timestamps
        if end_timestamp < start_timestamp:
            raise KeyError('End timestamp < Start timestamp', single_interval_info)
        if start_timestamp < full_r2f.first_valid_index():
            start_timestamp = full_r2f.first_valid_index()
        if end_timestamp > full_r2f.last_valid_index():
            end_timestamp = full_r2f.last_valid_index()

        # Extract the part of the case from the dataframe
        interval_part = full_r2f[start_timestamp: end_timestamp]
        label_interval_dfs[interval_id] = interval_part

    return label_interval_dfs


def convert_rul_to_interval(rul, rul_split: list):
    # Converts the precise rul value into a integer-representation of the corresponding interval.
    # Note: if the first entry in the split is 0, rul=0 will be considered as separate case.
    for interval, end in enumerate(rul_split):
        if rul <= end:
            rul_label = '_rul_' + str(int(end)) if end < np.inf else '_rul_max'
            return rul_label

    raise ValueError('This should not happen - rul_splits configured wrongly?')


def create_unique_label(r2f_row, config):
    # If this interval does not contain a failure the no_failure label is kept.
    if r2f_row['label'] == 'no_failure':
        return 'no_failure'
    else:

        # Base label: affected component combined with the type of the simulated failure.
        label = r2f_row['affected_component']

        if r2f_row['failure_mode'] != '':
            label += '_' + r2f_row['failure_mode']

        # If configured, also assign a remaining useful life interval.
        if label in config.rul_splits.keys():

            # With try block to also handle the case that the rul column is not present at all.
            try:
                row_rul = r2f_row['rul']

                if np.isnan(row_rul):
                    raise KeyError()

            except KeyError:
                raise KeyError(f'RUL split defined for label {label} but no RUL value given in run_to_failure_info!')

            # Map the stored value to one of the configured intervals and append the rul part to the base label.
            label += convert_rul_to_interval(row_rul, config.rul_splits.get(label))

        return label


def create_r2f_id(row, dataset_index):
    if row['label'] == 'no_failure':
        return f'nf_' + row['interval_id']
    else:
        return f'f_{dataset_index}_' + str(int(row['r2f_id']))


def split_interval_into_examples(config: Configuration, single_interval_df: pd.DataFrame):
    examples, next_values, example_start_times, example_end_times = [], [], [], []

    # Equal to the defined start and end points of the interval because of split_by_label functionality.
    start_time = single_interval_df.index[0]
    end_time = single_interval_df.index[-1]

    # time_series_length + 1 because split the result into actual examples and values of next timestamp.
    window_size = config.time_series_length + 1
    window_size_sec = (config.resample_frequency * config.time_series_length) / 1000
    resample_freq = str(config.resample_frequency) + 'ms'

    # Slide over data frame and extract windows until the window would exceed the last time step.
    examples_extracted_counter = 0
    while start_time + pd.to_timedelta(window_size_sec, unit='s') < end_time:
        extract_single_window(start_time, window_size, resample_freq, single_interval_df,
                              examples, next_values, example_start_times, example_end_times)

        # Update next start time for next window by adding the step size of the overlapping window.
        start_time = start_time + pd.to_timedelta(config.overlapping_window_step_seconds, unit='s')
        examples_extracted_counter += 1

    # Extract at least one example, even if failure interval does not cover a full example.
    if examples_extracted_counter == 0 and config.extract_single_example_short_intervals:
        # print('Extracted single example from short interval.')
        extract_single_window(start_time, window_size, resample_freq, single_interval_df,
                              examples, next_values, example_start_times, example_end_times)

    return examples, next_values, example_start_times, example_end_times


def extract_single_window(start_time, window_size, resample_freq, single_interval_df,
                          examples, next_values, example_start_times, example_end_times):
    # Generate a list with indexes for the current window.
    overlapping_window_indices = pd.date_range(start_time, periods=window_size, freq=resample_freq)

    example_df = single_interval_df.asof(overlapping_window_indices)
    example_array = example_df.to_numpy()

    # Split sampled values into actual example and values of next timestamp.
    example, next_values_example = example_array[0:-1], example_array[-1]
    examples.append(example)
    next_values.append(next_values_example)

    # -2 because last index corresponds to timestamp of next_values, not the actual example.
    example_start_times.append(example_df.index[0])
    example_end_times.append(example_df.index[-2])


def handle_interval(config, single_interval_info, single_interval_df):
    lists = split_interval_into_examples(config, single_interval_df)
    examples_interval, next_values_interval, start_times_examples_interval, end_times_examples_interval = lists
    number_examples = len(examples_interval)

    # Basic check that all parts are split up correctly.
    lengths = [len(l) for l in lists]
    assert all(x == lengths[0] for x in lengths), 'Problem during splitting of intervals into examples'

    rul_end_interval, failure_mode, start_interval, end_interval = np.nan, None, None, None

    # Repeat aux data as many times as examples were extracted from the interval.
    if number_examples == 0:
        single_interval_info = pd.DataFrame(columns=single_interval_info.index)
    elif number_examples == 1:
        single_interval_info = pd.DataFrame([single_interval_info])
    else:
        if 'rul' in single_interval_info.index:
            rul_end_interval, failure_mode = single_interval_info['rul'], single_interval_info['failure_mode']
            start_interval, end_interval = single_interval_info['start'], single_interval_info['end']

        single_interval_info = pd.concat([single_interval_info] * number_examples, axis=1).T

    # Use for debugging: Replaces examples values with identifier composed by interval and example id.
    # examples_interval = [interval + '--' + str(example_id) for interval, example_id in
    #                      zip(single_interval_info['interval_id'], range(number_examples))]

    # Interpolation of rul value per examples, len() check just be safe, should be covered by other condition.
    if len(single_interval_info) > 1 and not np.isnan(rul_end_interval):

        if rul_end_interval < 0.0:
            # RUL < 0.0 (explicitly -1.0 in config.json) means a total failure over the hole interval,
            # such that no interpolation should occur.
            single_interval_info['rul'] = -1.0
        elif 'linear' in failure_mode:
            # RUL drops linearly by one each second.
            interval_duration = (end_interval - start_interval).total_seconds()
            rul_start_interval = rul_end_interval + interval_duration
            rul_interpol = np.linspace(start=rul_start_interval, stop=rul_end_interval,
                                       num=number_examples, endpoint=True)
            single_interval_info['rul'] = rul_interpol
        else:
            print(f'RUL Interpolation not implemented for failure mode {failure_mode}! Static value is retained.')
            single_interval_info['rul'] = rul_end_interval

    # Replace the start and end times of the interval with the ones for each example.
    single_interval_info['start'] = start_times_examples_interval
    single_interval_info['end'] = end_times_examples_interval

    return single_interval_info, examples_interval, next_values_interval


def extract_examples(config: Configuration, dataset_name: str, dataset_index: int):
    r2f_info = config.run_to_failure_info.get(dataset_name)
    r2f_info = pd.DataFrame.from_dict(r2f_info)

    if len(r2f_info) == 0:
        print(f'Extracting dataset {dataset_index}: {dataset_name}\n')
        print(f'No valid intervals configured - Skipping example extraction ...')
        print('\n----------------------------------------------------------------------------')
        return

    #
    # Prepare run to failure information for this dataset
    #

    # Add an unique id to each interval.
    r2f_info['interval_id'] = str(dataset_index) + '_' + r2f_info.index.astype(str).values

    # Standardisation of missing values for further processing.
    r2f_info = r2f_info.fillna(value=np.nan)

    # Change label to unique failure class.
    r2f_info['label'] = r2f_info.apply(lambda row: create_unique_label(row, config), axis=1)

    # Edge case: When only no_failure entries are configured for this dataset this column may not exist.
    if 'failure_time' not in r2f_info.columns:
        r2f_info['failure_time'] = np.NaN

    # Add a run to failure id, corresponds to the interval id for no_failure or the failure time for failure intervals.
    r2f_info['r2f_id'] = pd.Categorical(r2f_info['failure_time'])
    r2f_info['r2f_id'] = r2f_info['r2f_id'].cat.codes
    r2f_info['r2f_id'] = r2f_info.apply(lambda row: create_r2f_id(row, dataset_index), axis=1)

    # Convert relevant columns to date time format.
    dt_cols = ['start', 'end', 'failure_time']
    for col in dt_cols:
        r2f_info[col] = pd.to_datetime(r2f_info[col], format=config.dt_format)

    print()
    print(f'Extracting dataset {dataset_index}: {dataset_name}\n')
    print(f'Run 2 Failure Information:')
    print(r2f_info.to_string())
    print('\n')

    # Load cleaned dataframe for this dataset.
    file_path = config.combined_and_cleaned_data_path + dataset_name + '.pkl'
    with open(file_path, 'rb') as file:
        full_r2f: pd.DataFrame = pd.read_pickle(file)

    # Split full dataframe into sub frames with a constant class label.
    label_interval_dfs: [pd.DataFrame] = split_by_label(full_r2f, r2f_info)

    examples: [np.ndarray] = []
    next_values: [np.ndarray] = []
    aux_df = pd.DataFrame()

    # Iterate over each interval with constant class label to extract the single examples.
    nbr_intervals = len(r2f_info)
    for interval in range(nbr_intervals):
        interval_info = r2f_info.loc[interval]
        interval_data = label_interval_dfs.get(interval_info['interval_id'])

        # print(f"Dataset {dataset_index}: Extracting {interval}/{nbr_intervals}: {interval_info['interval_id']}")

        if interval_data.shape[0] == 0:
            error = f'Dataset {dataset_index}: Configuration error in interval {interval}. ' \
                    f'Defined start or end time is outside of recorded time interval.'
            raise ValueError(error, interval_info['interval_id'])

        interval_info, examples_interval, next_values_interval = handle_interval(config, interval_info, interval_data)

        if examples_interval is not None and len(examples_interval) > 0:
            examples.extend(examples_interval)
            next_values.extend(next_values_interval)
            aux_df = aux_df.append(interval_info, ignore_index=True)

    # Convert lists of examples to numpy arrays.
    examples = np.stack(examples, axis=0)
    next_values = np.stack(next_values, axis=0)

    storage_path = config.extracted_examples_path + dataset_name + os.path.sep
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    # Store extracted examples for this dataset.
    np.save(storage_path + 'examples.npy', examples)
    np.save(storage_path + 'next_values.npy', next_values)
    aux_df.to_pickle(storage_path + 'aux_df.pkl')
    print(f"Dataset {dataset_index}: Finished extracting examples.")


def create_dataset(config, datasets):
    examples: [np.ndarray] = []
    next_values: [np.ndarray] = []
    aux_df = pd.DataFrame()

    # Load extracted examples and information for each dataset.
    for dataset_index, dataset_name in enumerate(datasets):

        storage_path = config.extracted_examples_path + dataset_name + os.path.sep

        # Check if folder for this dataset exists.
        if not os.path.exists(storage_path):
            print(f'Skipping dataset {dataset_index} {dataset_name} because no examples were extracted.')
            continue

        e = np.load(storage_path + 'examples.npy')
        n = np.load(storage_path + 'next_values.npy')
        a = pd.read_pickle(storage_path + 'aux_df.pkl')

        examples.append(e)
        next_values.append(n)
        aux_df = aux_df.append(a, ignore_index=True)

    # Convert lists of arrays into a single numpy array.
    x = np.concatenate(examples, axis=0)
    next_values = np.concatenate(next_values, axis=0)

    print('\nExamples before clean up:', x.shape[0], '\n')
    cleaner = DSCleaner(config, x, next_values, aux_df)
    cleaner.clean()
    x, next_values, aux_df = cleaner.return_all()
    print('\nExamples after clean up:', x.shape[0], '\n')

    check_stratify_problem(aux_df, limit=2, split_mode=config.split_mode)
    train_indices, test_indices = determine_train_test_indices(config, aux_df)
    train_val_indices, test_val_indices = [], []

    if config.train_val_split_size > 0:
        check_stratify_problem(aux_df, limit=2, main_indices=train_indices, split_mode=config.split_mode_val)
        train_indices, train_val_indices = determine_val_indices(config, aux_df, train_indices,
                                                                 config.train_val_split_size)

    if config.test_val_split_size > 0:
        check_stratify_problem(aux_df, limit=2, main_indices=test_indices, split_mode=config.split_mode_val)
        test_indices, test_val_indices = determine_val_indices(config, aux_df, test_indices, config.test_val_split_size)

    print('\nNormalising and storing dataset parts...\n')

    shuffle_indices(train_indices, train_val_indices, test_indices, test_val_indices, config.random_seed)

    normaliser = Normaliser()

    if not os.path.exists(config.get_training_data_path()):
        os.makedirs(config.get_training_data_path())

    handle_part(config, x, next_values, aux_df, train_indices, normaliser, dataset_part=DatasetPart.TRAIN)
    handle_part(config, x, next_values, aux_df, test_indices, normaliser, dataset_part=DatasetPart.TEST)
    handle_part(config, x, next_values, aux_df, train_val_indices, normaliser, dataset_part=DatasetPart.TRAIN_VAL)
    handle_part(config, x, next_values, aux_df, test_val_indices, normaliser, dataset_part=DatasetPart.TEST_VAL)

    normaliser.store_scalers(config.scaler_folder)
    print('\nDataset creation finished successfully...\n')


def check_stratify_problem(aux_df, limit, split_mode, main_indices=None):
    if split_mode != SplitMode.SEEDED_SK_LEARN:
        return

    # Necessary because the stratify parameter will raise an exception if only a single example is present for one class.
    error_cases = []
    y = aux_df['label'].values if main_indices is None else aux_df['label'].values[main_indices]

    labels, counts = np.unique(y, return_counts=True)
    for label, count in zip(labels, counts):
        if count < limit:
            error_cases.append(label)
    if len(error_cases) > 0:
        print('These classes will results in an error if SEEDED_SKLEARN is used because of the stratify parameter:')
        print(*error_cases, sep='\n')


def shuffle_indices(train_indices, train_val_indices, test_indices, test_val_indices, random_seed):
    rng = np.random.default_rng(seed=random_seed)
    rng.shuffle(train_indices)
    rng.shuffle(train_val_indices)
    rng.shuffle(test_indices)
    rng.shuffle(test_val_indices)


def handle_part(config: Configuration, x: np.ndarray, next_values: np.ndarray, aux_df: pd.DataFrame,
                indices, normaliser: Normaliser, dataset_part: DatasetPart):

    # The indices list is empty if no train / test validation set should be created.
    if len(indices) == 0:
        # Remove old files of this dataset part if present
        with contextlib.suppress(FileNotFoundError):
            os.remove(config.get_training_data_path(Representation.RAW) + f'x_{dataset_part.lower()}.npy')
            os.remove(config.get_training_data_path(Representation.RAW) + f'next_values_{dataset_part.lower()}.npy')
            os.remove(config.get_training_data_path(Representation.RAW) + f'aux_df_{dataset_part.lower()}.pkl')
        return

    # Reduce to examples of this dataset part.
    # Important: Index must be continuous in order to be able to correctly reference examples with it.
    x_part = x[indices, :, :]
    next_values_part = next_values[indices, :]
    aux_df_part = aux_df.loc[indices]
    aux_df_part = aux_df_part.reset_index(drop=True)

    if dataset_part == DatasetPart.TRAIN:
        x_part, next_values_part = normaliser.normalise_train(x_part, next_values_part)
    else:
        x_part, next_values_part = normaliser.normalise(x_part, next_values_part)

    x_part, next_values_part = x_part.astype('float32'), next_values_part.astype('float32')

    np.save(config.get_training_data_path(Representation.RAW) + f'x_{dataset_part.lower()}.npy', x_part)
    np.save(config.get_training_data_path(Representation.RAW) + f'next_values_{dataset_part.lower()}.npy',
            next_values_part)
    aux_df_part.to_pickle(config.get_training_data_path(Representation.RAW) + f'aux_df_{dataset_part.lower()}.pkl')

    print(dataset_part, x_part.shape, 'x')
    print(dataset_part, next_values_part.shape, 'next_values')
    print(dataset_part, aux_df_part.shape, 'aux_df\n')


def purge_old_files(config: Configuration):
    print('Deleting outdated files.')

    print('\nTraining data:\n')
    for rep in Rep.get_all(exclude=[Rep.RAW]):
        rep_path = config.get_training_data_path(representation=rep)
        purge_path(rep_path)


def purge_path(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.*"))
        for f in files:
            print(f'{f}')
            os.remove(f)


def main():
    config = Configuration()

    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_examples", default=False, action="store_true")
    args, _ = parser.parse_known_args()

    datasets = []
    for dataset_entry in config.stored_datasets:
        dataset_name: str = dataset_entry[0]
        datasets.append(dataset_name)

    ####################################################################################################################
    # First part: Extract single examples from the dataset dfs in parallel.
    ####################################################################################################################
    # extract_examples(config, datasets[4], 4)
    # sys.exit()

    if args.extract_examples:
        async_results = []

        with Pool(processes=config.multiprocessing_limit) as pool:

            for dataset_index, dataset_name in enumerate(datasets):
                result = pool.apply_async(extract_examples, (config, dataset_name, dataset_index))
                async_results.append(result)

                # Wait for a short time so that the initial output of the function is displayed correctly
                sleep(1)

            for result in async_results:
                _ = result.get()

        print('Extraction of examples finished.')
    else:
        if not os.path.isfile(config.extracted_examples_path + datasets[0] + os.path.sep + 'examples.npy'):
            raise FileNotFoundError('Trying to create a dataset without examples being extracted.'
                                    ' Use --extract_examples to do so before creating the dataset.')

    ####################################################################################################################
    # Second part: Combine examples into a dataset.
    ####################################################################################################################
    create_dataset(config, datasets)

    # Save feature name array again in case "someone" deleted it
    file_name = config.combined_and_cleaned_data_path + datasets[0] + '.pkl'
    df: pd.DataFrame = pd.read_pickle(file_name)
    feature_names = np.array(list(df.columns))

    np.save(config.get_training_data_path(Representation.RAW) + 'feature_names.npy', feature_names)

    purge_old_files(config)


if __name__ == '__main__':
    main()
