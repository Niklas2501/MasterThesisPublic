import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from configuration.Configuration import Configuration


class DSCleaner:

    def __init__(self, config: Configuration, examples: np.ndarray, next_values: np.ndarray, aux_df: pd.DataFrame):
        self.config = config
        self.examples = examples
        self.next_values = next_values
        self.aux_df = aux_df

    def clean(self):
        # Must be executed in this order to ensure labels in data matches the ones defined in cleaning_info.
        self.rename_labels()
        self.remove_examples_by_label()
        self.clean_aux_df()

    def rename_labels(self):

        # Renaming of label in aux_df should be obsolete but kept to be safe.
        # To be safe, also apply renaming to unused labels list such that remove_examples_by_label works correctly.
        for key, value in self.config.label_renaming_mapping.items():
            self.aux_df['label'] = self.aux_df['label'].str.replace(key, value)

            # if len(self.config.unused_labels) > 0:
            #     self.config.unused_labels = np.char.replace(self.config.unused_labels, key, value)
            # if len(self.config.labels_used) > 0:
            #     self.config.labels_used = np.char.replace(self.config.labels_used, key, value)

    def remove_examples_by_label(self):
        indices_to_remove = []

        if len(self.config.labels_used) > 0:
            labels_present = np.unique(self.aux_df['label'].values.astype('str'))
            unused_labels = [label for label in labels_present if label not in self.config.labels_used]
        else:
            unused_labels = self.config.unused_labels

        for label in unused_labels:
            indices_with_label = self.aux_df.index[self.aux_df['label'] == label].tolist()
            indices_to_remove.extend(indices_with_label)

            if len(indices_with_label) > 0:
                print(f'\t{len(indices_with_label)} examples removed with label {label}')

        indices_remaining = list(set(range(len(self.examples))).difference(set(indices_to_remove)))

        self.aux_df = self.aux_df.loc[indices_remaining]
        self.examples = self.examples[indices_remaining]
        self.next_values = self.next_values[indices_remaining]

        # Important: Index must be continuous in order to be able to  correctly reference examples with it.
        self.aux_df = self.aux_df.reset_index(drop=True)

    def return_all(self):
        return self.examples, self.next_values, self.aux_df

    def clean_aux_df(self):
        for target_type, cols in self.config.aux_df_type_mapping.items():

            if target_type == 'numeric':
                for col in cols:

                    if col not in self.aux_df.columns:
                        continue

                    self.aux_df[col] = pd.to_numeric(self.aux_df[col])
            elif target_type == 'datetime':
                for col in cols:

                    if col not in self.aux_df.columns:
                        continue

                    self.aux_df[col] = pd.to_datetime(self.aux_df[col], format=self.config.dt_format)
            elif target_type == 'str':
                if len(cols) > 0:
                    print("Dataframe columns can not be converted to a string datatype. Must handled during import.")
            else:
                for col in cols:
                    self.aux_df[col] = self.aux_df[col].values.astype(target_type)


class Normaliser:

    def __init__(self):
        self.called_train = False
        self.scalers = None

    def normalise_train(self, x_train, next_values_train):
        nbr_of_features = x_train.shape[2]
        self.scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(nbr_of_features)]

        x_train = self.__execute(x_train, fit=True)
        next_values_train = self.__execute_for_next(next_values_train)

        self.called_train = True
        return x_train, next_values_train

    def normalise(self, array: np.ndarray, next_values: np.ndarray):
        if not self.called_train:
            raise Exception('normalise_train has be called once before.')

        array = self.__execute(array, fit=False)
        next_values = self.__execute_for_next(next_values)

        return array, next_values

    def __execute(self, array, fit=False):
        input_array = array.copy()

        if fit:
            print(f'Fitting normaliser to train with shape {array.shape} ...')
        else:
            print(f'Transforming array with shape {array.shape} ...')

        for feature_index, scaler in enumerate(self.scalers):
            # reshape column vector over each example and timestamp to a flatt array
            # necessary for normalisation to work properly
            shape_before = array[:, :, feature_index].shape
            array_shaped = array[:, :, feature_index].reshape(shape_before[0] * shape_before[1], 1)

            if fit:
                array_shaped = scaler.fit_transform(array_shaped)
            else:
                array_shaped = scaler.transform(array_shaped)

            # reshape back to original shape and assign normalised values
            array[:, :, feature_index] = array_shaped.reshape(shape_before)

        # For debugging purposes: Interleaves features with and without normalisation
        # In order for visualisation to work, features_names.npy must also be modified
        # C = np.empty((input_array.shape[0], input_array.shape[1], input_array.shape[2] * 2))
        # C[:, :, ::2] = input_array
        # C[:, :, 1::2] = array
        # array = C

        return array

    def __execute_for_next(self, next_values):

        for feature_index, scaler in enumerate(self.scalers):
            shape_before = next_values[:, feature_index].shape
            next_values_shaped = next_values[:, feature_index].reshape(shape_before[0], 1)
            next_values_shaped = scaler.transform(next_values_shaped)
            next_values[:, feature_index] = next_values_shaped.reshape(shape_before)

        return next_values

    def store_scalers(self, path):
        for i, scaler in enumerate(self.scalers):
            scaler_filename = path + 'scaler_' + str(i) + '.save'
            joblib.dump(scaler, scaler_filename)
