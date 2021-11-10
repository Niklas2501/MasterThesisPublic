import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from stgcn.Dataset import Dataset


def unique_value_counter(config: Configuration, dataset: Dataset, limit=None):
    x_train = dataset.get_train().get_x()
    feature_names = dataset.get_feature_names()

    print('Input shape: examples x timestamps x features: ', x_train.shape)

    # change to examples x features x timestamps
    x_train = np.swapaxes(x_train, 1, 2)
    x_train = np.swapaxes(x_train, 0, 1)
    x_train = np.reshape(x_train, newshape=(x_train.shape[0], -1))

    print('Prepared shape: features x values at timestamps for all examples: ', x_train.shape)

    df = pd.DataFrame(x_train.T, columns=feature_names)

    print('Calculating describe overview and writing to excel ...\n')
    df.describe(percentiles=[.05, .25, .50, .75, .95]).to_excel('../logs/value_overview.xlsx')

    for col in df.columns:
        col_values = df[col].values
        mean = np.mean(col_values)
        mean_abs_diff = np.mean(np.absolute(col_values - mean))
        mean_sqr_diff = np.mean(np.square(col_values - mean))

        values, counts = np.unique(col_values, return_counts=True)

        col_df = pd.DataFrame({'Values': values, 'Counts': counts})
        col_df = col_df.sort_values(by=['Counts'], ascending=False)

        print(col)
        print()
        print('Mean Absolute Deviation', mean_abs_diff)
        print('Mean Squared Deviation', mean_sqr_diff)
        print('STD', np.std(col_values))
        print()
        print(col_df.head(limit).to_string(index=False, show_dimensions=True, max_colwidth=15, col_space=15))
        print()
        print()


def list_of_streams(config: Configuration):
    outer_dict = {'txt_topics.pkl': {}, 'sensor_topics.pkl': {}, 'webserver_topics.pkl': {},
                  'failure_sim_topics.pkl': {}}

    outer_dict.pop('webserver_topics.pkl')
    outer_dict.pop('failure_sim_topics.pkl')
    unique_feature_names = []

    for dataset_entry in config.stored_datasets:
        dataset_name: str = dataset_entry[0]
        dataset_name = dataset_name if dataset_name.endswith('/') else dataset_name + '/'

        dfs_path = config.unprocessed_data_path + dataset_name

        for file in outer_dict.keys():
            df: pd.DataFrame = pd.read_pickle(dfs_path + file)

            values = {}
            for feature in list(df.columns):
                values[feature] = 1

            outer_dict[file][dataset_name] = values

    x = pd.DataFrame(outer_dict)

    for col in x.columns:
        d_list = x[col].values
        sets = x.index

        new_entries_dict = {}

        for c, values in zip(sets, d_list):
            new_entries_dict[c.replace('/', '')] = values

        y = pd.DataFrame(new_entries_dict)
        y = y.fillna(value=0)
        name = col.split('.')[0] + '_feature_overview.xlsx'
        y.to_excel(config.additional_files_folder + name)

        unique_feature_names.extend(list(y.index.values))

    unique_feature_names = list(set(unique_feature_names))
    unique_feature_names = sorted(unique_feature_names)

    with open(config.additional_files_folder + "unique_features.txt", "w") as file:

        for element in unique_feature_names:
            file.write(element + "\n")

    with open(config.additional_files_folder + "unique_features.json", "w", encoding='utf-8') as file:
        json.dump(unique_feature_names, file, ensure_ascii=False, indent=4)

    print('Unique feature count:', len(unique_feature_names))
    print('Files saved to: ', config.additional_files_folder)


if __name__ == '__main__':
    config = Configuration()
    dataset = Dataset(config)
    dataset.load()

    unique_value_counter(config, dataset, 20)
    # list_of_streams(config)
