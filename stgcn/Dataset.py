from threading import Thread

import numpy as np
import pandas as pd
from sklearn import preprocessing

from configuration.Configuration import Configuration
from configuration.Enums import Dataset as DatasetEnum
from configuration.Enums import DatasetPart as DPE
from configuration.Enums import Representation, LossFunction


class DatasetPart:

    def __init__(self, type: DPE, part_of_third_party):
        self.__x = None  # training data (examples, time steps, channels)
        self.__aux_df = None

        self.__y = None
        self.__y_strings = None
        self.__y_strings_unique = None
        self.__start_times = None
        self.__end_times = None
        self.__failure_times = None

        self.__num_instances = 0
        self.__type: DPE = type
        self.__part_of_third_party = part_of_third_party
        self.__loaded = False

        self.class_index_to_examples = {}

    def load_from_files(self, folder_path):
        descriptor = self.get_type().lower()

        try:
            self.__x = np.load(folder_path + f'x_{descriptor}.npy')  #
            self.__aux_df = pd.read_pickle(folder_path + f'aux_df_{descriptor}.pkl')

            # Mapping to vars
            self.__y_strings = self.__aux_df['label'].values.astype('str')
            self.__y_strings_unique = np.unique(self.__y_strings)

            f = '%Y-%m-%d %H:%M:%S.%f'
            self.__failure_times = self.__aux_df['failure_time'].dt.strftime(f).values.astype('str')
            self.__start_times = self.__aux_df['start'].dt.strftime(f).values.astype('str')
            self.__end_times = self.__aux_df['end'].dt.strftime(f).values.astype('str')
        except FileNotFoundError:
            return False

        self.__loaded = True
        self.__num_instances = self.get_x().shape[0]

        return True

    def encode_labels(self, encoder):
        if not self.__loaded:
            return

        # np.expand_dims(self.__y_strings, axis=-1) for one hot encoding
        # self.__y_strings for int encoding
        y_strings_shaped = np.expand_dims(self.__y_strings, axis=-1)
        self.__y = encoder.transform(y_strings_shaped)

    def get_x(self):
        return self.__x

    def apply_feature_mask(self, mask: np.ndarray):
        self.__x = self.__x[:, :, mask]

    def get_aux_df(self):
        return self.__aux_df

    def get_y(self):
        return self.__y

    def get_y_strings(self):
        return self.__y_strings

    def get_y_strings_unique(self):
        return self.__y_strings_unique

    def get_start_times(self):
        return self.__start_times

    def get_end_times(self):
        return self.__end_times

    def get_start_time(self, index):
        return self.get_start_times()[index]

    def get_end_time(self, index):
        return self.get_end_times()[index]

    def get_time_window_str(self, index):
        t0 = self.get_start_time(index)
        t1 = self.get_end_time(index)
        # Comment both next lines if date should also be printed, else only the time component is returned
        # rep = lambda x: x.split(' ')[-1]
        # t0, t1 = rep(t0), rep(t1)

        return " - ".join([t0, t1])

    def get_failure_times(self):
        return self.__failure_times

    def get_num_instances(self):
        return self.__num_instances

    def get_type(self):
        return self.__type

    def is_part_of_third_party(self):
        return self.__part_of_third_party

    def get_indices_with_label(self, label):
        return np.where(self.get_y_strings() == label)[0]

    def get_failure_indices(self):
        return np.where(self.get_y_strings() != 'no_failure')[0]

    def get_failure_time(self, index):
        return self.get_failure_times()[index]

    def get_true_label(self, index):
        return self.get_y()[index]


class Dataset:

    def __init__(self, config: Configuration):
        super().__init__()
        self.config: Configuration = config
        self.is_third_party_dataset = DatasetEnum.is_3rd_party(config.dataset)

        self.__train = DatasetPart(DPE.TRAIN, self.is_third_party_dataset)
        self.__test = DatasetPart(DPE.TEST, self.is_third_party_dataset)
        self.__train_val = DatasetPart(DPE.TRAIN_VAL, self.is_third_party_dataset)
        self.__test_val = DatasetPart(DPE.TEST_VAL, self.is_third_party_dataset)

        self.time_series_length = None
        self.time_series_depth = None

        # The names of all features in the dataset loaded from files.
        self.feature_names = None

        # The total number of classes.
        self.num_classes = None

        # Array that contains all classes that occur in any of the dataset parts.
        self.unique_labels_overall = None

        self.__has_train_val = None
        self.__has_test_val = None
        self.num_instances = 0

        self.label_encoder = None

        self.is_wip_dataset = DatasetEnum.is_wip_dataset(self.config.dataset)
        self.is_feature_vector_dataset = DatasetEnum.is_feature_vector_dataset(config.dataset) \
                                         or Representation.contains_feature_vectors(config.representation)

    def load(self, print_info=True):
        """
        Load the dataset from the configured location
        :param print_info: Output of basic information about the dataset at the end
        """

        self.feature_names = np.load(self.config.get_training_data_path() + 'feature_names.npy')
        self.__train.load_from_files(self.config.get_training_data_path())
        self.__test.load_from_files(self.config.get_training_data_path())
        self.__has_train_val = self.__train_val.load_from_files(self.config.get_training_data_path())
        self.__has_test_val = self.__test_val.load_from_files(self.config.get_training_data_path())

        parts = [self.__train, self.__test]

        if self.has_test_val():
            parts.append(self.__test_val)

        if self.has_train_val():
            parts.append(self.__train_val)

        self.unique_labels_overall = []
        for part in parts:
            self.unique_labels_overall.append(part.get_y_strings_unique())
            self.num_instances += part.get_num_instances()

        self.unique_labels_overall = np.unique(np.concatenate(self.unique_labels_overall))
        self.num_classes = self.unique_labels_overall.size

        self.label_encoder = preprocessing.OneHotEncoder(sparse=False)  # LabelEncoder()

        # .fit(np.expand_dims(self.unique_labels_overall, axis=-1)) for one hot
        # .fit(self.unique_labels_overall) for sparse
        self.label_encoder = self.label_encoder.fit(np.expand_dims(self.unique_labels_overall, axis=-1))

        for part in parts:
            part.encode_labels(self.label_encoder)

        self.subsequent_feature_reduction(parts)

        # Length of the second array dimension is the length of the time series.
        self.time_series_length = self.__train.get_x().shape[1]

        # Length of the third array dimension is the number of channels = (independent) readings at this point of time.
        self.time_series_depth = self.__train.get_x().shape[2] if not self.is_feature_vector_dataset else 1

        if print_info:
            print()
            print(f'Dataset in {self.config.get_training_data_path()} loaded:')
            print('Shape of training set (example, time, channels):', self.get_train().get_x().shape)
            print('Shape of test set (example, time, channels):', self.get_test().get_x().shape)
            if self.has_train_val():
                print('Shape of train validation set (example, time, channels):', self.get_train_val().get_x().shape)
            if self.has_test_val():
                print('Shape of test validation set (example, time, channels):', self.get_test_val().get_x().shape)
            print('Num of classes in all:', self.num_classes)
            print()

    @staticmethod
    def apply_part_functions(functions, args):
        threads = []
        for function in functions:
            thread = Thread(target=function, args=args)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def create_feature_mask(self, features_included_in_mask, return_excluded=False):
        mask = np.zeros(self.feature_names.size, dtype=int)
        features_excluded = []

        for feature_index, feature_name in enumerate(self.feature_names):
            if feature_name in features_included_in_mask:
                mask[feature_index] = 1
            else:
                features_excluded.append(feature_name)

        mask = mask.astype(bool)

        if return_excluded:
            return mask, features_excluded
        else:
            return mask

    def subsequent_feature_reduction(self, parts: [DatasetPart]):
        features_configured = np.array(sorted(self.config.features_used))

        if not self.config.subsequent_feature_reduction or np.array_equal(features_configured, self.feature_names):
            return
        else:
            print(f'Subsequent reduction from {self.feature_names.size} to {features_configured.size} features.')

        missing_features = [feature for feature in features_configured if feature not in self.feature_names]
        if len(missing_features) > 0:
            raise ValueError(f'Feature adjustment is not possible because these configured features are missing'
                             f' in the dataset: {missing_features}')

        print('\tReducing features in dataset. Removed:')

        mask, features_excluded = self.create_feature_mask(features_configured, return_excluded=True)
        for f in features_excluded:
            print(f'\t\t{f}')

        self.feature_names = self.feature_names[mask]

        self.apply_part_functions([part.apply_feature_mask for part in parts], args=(mask,))
        self.regenerate_feature_dependent_files()

    def regenerate_feature_dependent_files(self):
        from analytic_tools.Onto2Matrix import onto_2_matrix
        from analytic_tools.GenerateEmbeddings import generate_embeddings
        temp_id = np.random.default_rng().integers(1000, 9999)

        print('\tRegenerate predefined matrix ...')
        onto_2_matrix(self.config, self.feature_names, daemon=True, temp_id=temp_id)
        print('\tRegenerate embeddings file ...')
        generate_embeddings(self.config, self.feature_names, daemon=True, temp_id=temp_id)
        print()

    def get_part(self, dataset_type):
        if dataset_type == DPE.TRAIN:
            return self.get_train()
        if dataset_type == DPE.TEST:
            return self.get_test()
        if dataset_type == DPE.TRAIN_VAL:
            return self.get_train_val()
        if dataset_type == DPE.TEST_VAL:
            return self.get_test_val()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def has_test_val(self):
        return self.__has_test_val

    def has_train_val(self):
        return self.__has_train_val

    def get_train(self):
        return self.__train

    def get_test(self):
        return self.__test

    def get_train_val(self):
        if self.has_train_val():
            return self.__train_val
        else:
            raise ValueError('This dataset does not have a train validation set.')

    def get_test_val(self):
        if self.has_test_val():
            return self.__test_val
        else:
            raise ValueError('This dataset does not have a test validation set.')

    def get_feature_names(self):
        return self.feature_names

    def get_class_weights(self, variant):
        wcce_v3_nf_weight, wcce_v4_f_weight = 0.75, 2.0
        train_labels = self.get_train().get_y_strings()
        unique_labels, positive = np.unique(train_labels, return_counts=True)
        negative = train_labels.shape[0] - positive

        assert np.array_equal(unique_labels, self.unique_labels_overall), \
            'Unique labels for used for weight calculation do not match the ones used for one hot encoding.' \
            'This would result in a wrong calculation.'

        # V1 and V2 based on https://doi.org/10.3390/s20226433
        if variant == LossFunction.WCCE_V1:
            weights = (positive + negative) / positive
        elif variant == LossFunction.WCCE_V2:
            weights = negative / (positive + negative)
        elif variant == LossFunction.WCCE_V3:
            weights = np.ones((self.unique_labels_overall.shape[0]))
            weights[0] = wcce_v3_nf_weight
        elif variant == LossFunction.WCCE_V4:
            weights = np.full((self.unique_labels_overall.shape[0]), fill_value=wcce_v4_f_weight)
            weights[0] = 1
        else:
            raise ValueError('Unknown weight variant.')

        return weights
