import argparse
import os
import sys
from shutil import copyfile
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from sktime.transformations.panel.rocket import Rocket, MiniRocketMultivariate

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Enums import Representation as RepEnum
from configuration.Configuration import Configuration
from stgcn.Dataset import Dataset


class RocketRepresentation:

    def __init__(self, config: Configuration, dataset: Dataset):
        self.config: Configuration = config
        self.dataset: Dataset = dataset
        self.x_train_features = None
        self.x_test_features = None
        self.x_train_val_features = None
        self.x_test_val_features = None

    def create_representation(self, variant):

        if variant == 'standard':
            rep_enum = RepEnum.ROCKET
            rocket = Rocket(num_kernels=self.config.rocket_kernels,
                            normalise=False, random_state=self.config.random_seed)
        elif variant == 'mini':
            rep_enum = RepEnum.MINI_ROCKET
            rocket = MiniRocketMultivariate(num_features=self.config.rocket_kernels,
                                            random_state=self.config.random_seed)
        else:
            raise ValueError('Invalid rocket variant specified.')

        x_train, x_test = self.dataset.get_train().get_x(), self.dataset.get_test().get_x()

        # Cast is necessary because rocket seems to expect 64 bit values
        x_train_casted = x_train.astype('float64')
        x_test_casted = x_test.astype('float64')

        print('Transforming from array to dataframe...')
        x_train_df = self.array_to_ts_df(x_train_casted)
        x_test_df = self.array_to_ts_df(x_test_casted)

        print('Started fitting ...')
        start = perf_counter()
        rocket.fit(x_train_df)
        end = perf_counter()

        print(f'Finished fitting. Duration: {round(end - start, 3)} seconds.\n')

        print('Transforming train dataset...')
        start = perf_counter()
        x_train = rocket.transform(x_train_df).values
        end = perf_counter()
        print(f'Transforming train dataset finished. Duration: {round(end - start, 3)} seconds.\n')

        print('Transforming test dataset...')
        start = perf_counter()
        x_test = rocket.transform(x_test_df).values
        end = perf_counter()
        print(f'Transforming test dataset finished. Duration: {round(end - start, 3)} seconds.\n')

        # Convert back into datatype that is equivalent to the one used by the neural net
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        rocket_folder = self.config.get_training_data_path(representation=rep_enum)
        raw_folder = self.config.get_training_data_path(representation=RepEnum.RAW)

        np.save(rocket_folder + 'x_train.npy', x_train)
        np.save(rocket_folder + 'x_test.npy', x_test)

        # Save array with placeholder feature names for compatibility reasons
        placeholder = ['feature_' + str(i) for i in range(x_train.shape[1])]
        placeholder = np.array(placeholder)
        np.save(rocket_folder + 'feature_names.npy', placeholder)

        if self.dataset.has_train_val():
            print('Transforming train validation dataset...')
            x_train_val = self.dataset.get_train_val().get_x()
            x_train_val_casted = x_train_val.astype('float64')
            x_train_val_df = self.array_to_ts_df(x_train_val_casted)
            start = perf_counter()
            x_train_val = rocket.transform(x_train_val_df).values
            end = perf_counter()
            print(f'Transforming train validation dataset finished. Duration: {round(end - start, 3)} seconds.\n')
            x_train_val = x_train_val.astype('float32')
            np.save(rocket_folder + 'x_train_val.npy', x_train_val)

        if self.dataset.has_test_val():
            print('Transforming test validation dataset...')
            x_test_val = self.dataset.get_test_val().get_x()
            x_test_val_casted = x_test_val.astype('float64')
            x_test_val_df = self.array_to_ts_df(x_test_val_casted)
            start = perf_counter()
            x_test_val = rocket.transform(x_test_val_df).values
            end = perf_counter()
            print(f'Transforming test validation dataset finished. Duration: {round(end - start, 3)} seconds.\n')
            x_test_val = x_test_val.astype('float32')
            np.save(rocket_folder + 'x_test_val.npy', x_test_val)

        print('\nRepresentation generation finished.')
        print('Resulting shape train: ', x_train.shape)
        print('Resulting shape test: ', x_test.shape)

        if self.dataset.has_train_val():
            # noinspection PyUnboundLocalVariable
            print('Resulting shape train val: ', x_train_val.shape)

        if self.dataset.has_test_val():
            # noinspection PyUnboundLocalVariable
            print('Resulting shape test val: ', x_test_val.shape)

        self.copy_files(raw_folder, rocket_folder)

        print('\nRepresentation creation finished.')

    # Numpy dataset must be converted to expected format described
    # @ https://www.sktime.org/en/latest/examples/loading_data.html
    @staticmethod
    def array_to_ts_df(array):
        # Input : (Example, Timestamp, Feature)
        # Temp 1: (Example, Feature, Timestamp)
        array_transformed = np.einsum('abc->acb', array)

        # No simpler / more elegant solution via numpy or pandas found
        # Create list of examples with list of features containing a pandas series of  timestamp values
        # Temp 2: (Example, Feature, Series of timestamp values)
        list_of_examples = []

        for example in array_transformed:
            ex = []
            for feature in example:
                ex.append(pd.Series(feature))

            list_of_examples.append(ex)

        # Conversion to dataframe with expected format
        return pd.DataFrame(data=list_of_examples)

    @staticmethod
    def reshape(array):
        return np.expand_dims(array, axis=2)

    @staticmethod
    def copy_files(source_folder, target_folder):
        changed_files = ['feature_names.npy', 'x_train.npy', 'x_test.npy',
                         'x_test_val.npy', 'x_train_val.npy']

        source_folder = source_folder if source_folder.endswith('/') else source_folder + '/'
        target_folder = target_folder if target_folder.endswith('/') else target_folder + '/'

        print('\nCopying additional files to the representation directory:')
        for file in os.listdir(source_folder):
            if file not in changed_files:
                print('\tCopying ' + file + '...')
                copyfile(source_folder + file, target_folder + file)


def main():
    config = Configuration()

    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',
                        choices=['standard', 'mini'],
                        required=True,
                        help='Rocket variant that should be used.')
    args, _ = parser.parse_known_args()

    assert config.representation == RepEnum.RAW, 'config.representation must be set to Representation.RAW'

    p = psutil.Process()
    cores = p.cpu_affinity()
    p.cpu_affinity(cores[0:config.multiprocessing_limit])

    dataset = Dataset(config)
    dataset.load()

    r = RocketRepresentation(config, dataset)
    r.create_representation(args.variant)


if __name__ == '__main__':
    main()
