import json

import numpy as np

from configuration.Enums import *


####
# Note: Division into different classes only serves to improve clarity.
# Only the Configuration class should be used to access all variables.
# Important: It must be ensured that variable names are only used once.
# Otherwise they will be overwritten depending on the order of inheritance!
# All methods should be added to the Configuration class to be able to access all variables
####


class GeneralConfiguration:

    def __init__(self):
        ###
        # Overall settings
        ###

        # Specifies the maximum number of cores to be used when using multiprocessing
        self.multiprocessing_limit = 25

        # Select whether tensorflow should allocate the full gpu memory (False) or if growing memory allocation is used.
        self.tf_memory_growth_enabled = False

        # Path and file name to the specific model that should be used for testing.
        # Folder where the models are stored is prepended below.
        self.filename_model_to_use = 'stgcn_gcn_g_a_emb_v1_25011997'

        self.dataset = Dataset.FT_MAIN
        self.representation = Representation.RAW

        ##
        # Debugging / Testing - Don't use for long term feature implementation
        ##


class TrainingConfiguration:

    def __init__(self):
        ###
        # Hyperparameters
        ###

        # Main directory where the hyperparameter config files are stored
        self.hyper_file_folder = '../configuration/hyperparameter_combinations/'

        # The following setting is ignored if grid search is used
        self.hyper_file = self.hyper_file_folder + '_selected_models/stgcn_gcn_g_a_emb_v1.json'

        self.grid_search_mode = GridSearchMode.TEST_MULTIPLE_JSONS
        self.grid_search_config_file = '_selected_models.json'
        self.grid_search_base_template_file = ''  # self.grid_search_config_file

        # Selection whether checkpoints should be created during the training of models.
        self.create_checkpoints = False

        # If true, the folder with temporary models created during training is deleted after final model was stored.
        # Only relevant when creat_checkpoints is True.
        self.delete_temp_models_after_training = True

        # Selection whether a secondary output of the model based on train validation data as input should be stored.
        self.write_vis_out_to_file = False
        self.write_a_out_to_file = True

        # Selection whether plots of the adjacency matrix used are created at the end of training and
        # whether these plots include the feature names.
        self.create_a_plots = True
        self.a_plot_display_labels = True

        # Determines whether the results of a single model are printed during grid search.
        self.output_intermediate_results = True

        # Disables the storage of models trained during grid search. This also overwrites create_a_plots.
        self.do_not_save_gs_models = False

        #
        # Testing
        #

        # Selection of the dataset part on which the evaluation is executed
        self.part_tested = DatasetPart.TEST

        # Whether only examples that represent a failure (label != 'no_failure') are tested in the evaluation
        self.test_failures_only = False

        # Reduction of dataset to configured features at run time. Should primarily only be used for feature selection
        self.subsequent_feature_reduction = False

        #
        # Baseline
        #

        self.baseline_algorithm = BaselineAlgorithm.RIDGE_CLASSIFIER

        self.rocket_kernels = 10_000  # Default value recommended for rocket.


class PreprocessingConfiguration:

    def __init__(self):
        ###
        # This configuration contains information and settings relevant for the data preprocessing and dataset creation
        ###

        # Value is used to ensure a constant frequency of the measurement time points
        # Needs to be the same for DataImport as well as DatasetCreation
        # Unit: ms
        self.resample_frequency = 8

        # Define the length (= the number of timestamps) of the time series generated
        self.time_series_length = 500

        # Defines the step size of window, e.g. for = 2: Example 0 starts at 00:00:00 and Example 1 at 00:00:02
        # For no overlapping: value = seconds(time_series_length * resample frequency)
        self.overlapping_window_step_seconds = 1

        # Needs to be the same for DataImport as well as DatasetCreation
        self.create_secondary_feature_set = False

        # Select whether a single failure example should be extracted from intervals that is shorter than the
        # window size defined by the frequency and the time series length.
        self.extract_single_example_short_intervals = False

        # Select if first column should be dropped when converting categorical columns into a one hot representation.
        # Recommended for neural networks.
        self.drop_first_one_hot_column = True

        # Share of examples used as test set.
        self.test_split_size = 0.35

        # Custom share for the no failure class for the train/test split.
        self.use_separate_nf_split_size = True
        self.nf_test_split_size = 0.15

        # Share of examples which is separated from the test set to be used as validation set.
        self.test_val_split_size = 0.50

        # Share of examples which is separated from the train set to be used as validation set.
        self.train_val_split_size = 0.15

        # Defines the method how examples are split into train and test, see enum class for details.
        self.split_mode = SplitMode.ENSURE_NO_MIX
        self.split_mode_val = SplitMode.SEEDED_SK_LEARN


class StaticConfiguration:

    # noinspection PyUnresolvedReferences
    def __init__(self):
        ###
        # This configuration contains data that rarely needs to be changed, such as the paths to certain directories
        ###

        self.kafka_server_ip = '192.168.1.111:9092'
        self.dt_format = '%Y-%m-%d %H:%M:%S.%f'
        self.use_hyper_file = True

        # seed for various tasks that involve some kind of random selection
        self.random_seed = 25011997

        ##
        # All of the following None-Variables are read from the config.json file because they are mostly static
        # and don't have to be changed very often
        self.kafka_topic_names = None
        self.datasets_to_import_from_kafka = None
        self.stored_datasets = None
        self.run_to_failure_info = None
        self.aux_df_type_mapping = None

        self.float_features = None
        self.integer_features = None
        self.categorical_features = None
        self.primary_features = None
        self.secondary_features = None
        self.categorical_nan_replacement = None
        self.features_used = None

        # mapping for topic name to prefix of sensor streams, relevant to get the same order of streams
        self.prefixes = None

        self.case_to_individual_features = None
        self.case_to_individual_features_strict = None

        self.label_renaming_mapping = None
        self.one_hot_features_to_remove = None
        self.transfer_from_train_to_test = None
        self.labels_used = None
        self.unused_labels = None
        self.rul_splits = None

        self.kafka_webserver_topics = None
        self.kafka_sensor_topics = None
        self.kafka_txt_topics = None
        self.kafka_failure_sim_topics = None

        if self.dataset == Dataset.FT_MAIN:
            self.load_config_json('../configuration/config.json')
        elif self.dataset == Dataset.FT_LEGACY:
            self.load_config_json('../configuration/config_legacy_new.json')
        else:
            self.load_config_json('../configuration/config.json')

        self.grid_search_configs_folder = '../configuration/grid_search_configs/'
        self.grid_search_base_templates_folder = '../configuration/grid_search_base_templates/'
        self.grid_search_config = {}

        # noinspection PyUnresolvedReferences
        self.load_grid_search_files()

        ##
        # Folders and file names
        ##
        # Note: Folder of used model specified in GeneralConfiguration
        self.base_path = '../data/'

        # Folder where the trained models are saved to during learning process
        self.models_folder = self.get_dataset_path() + 'trained_models/'

        # noinspection PyUnresolvedReferences
        self.directory_model_to_use = self.models_folder + self.filename_model_to_use + '/'

        # Folder where the preprocessed training and test data for the neural network should be stored
        self.training_data_folder = self.get_training_data_path()

        # Folder where the normalisation models should be stored
        self.scaler_folder = self.get_dataset_path() + 'scaler/'

        self.additional_files_folder = self.get_dataset_path() + 'additional_files/'

        self.kafka_imported_topics_path = self.get_dataset_path() + 'datasets/kafka_import/'
        self.unprocessed_data_path = self.get_dataset_path() + 'datasets/unprocessed_data/'
        self.combined_and_cleaned_data_path = self.get_dataset_path() + 'datasets/combined_and_cleaned_datasets/'
        self.extracted_examples_path = self.get_dataset_path() + 'datasets/extracted_examples/'

        self.a_pre_file = 'a_pre.xlsx'
        self.embeddings_file = 'embeddings.npy'

    # noinspection PyUnresolvedReferences
    def get_dataset_path(self):

        if self.dataset == Dataset.FT_MAIN:
            relative_path = ''
        elif self.dataset == Dataset.FT_LEGACY:
            relative_path = 'other_ft_versions/FT_LEGACY/'
        else:
            raise ValueError('Unknown dataset:', self.dataset)

        return self.base_path + relative_path

    # noinspection PyUnresolvedReferences
    def get_training_data_path(self, representation=None):
        path = self.get_dataset_path() + 'training_data/'
        representation = representation if representation is not None else self.representation

        if representation == Representation.RAW:
            path += 'raw_data/'
        elif representation == Representation.ROCKET:
            path += 'rocket_rep/'
        elif representation == Representation.MINI_ROCKET:
            path += 'mini_rocket_rep/'
        else:
            raise ValueError('Unknown representation:', self.representation)

        return path

    def get_additional_data_path(self, file_name):
        path = self.get_dataset_path() + 'training_data/'
        path += 'additional_data/' + file_name

        return path


class Configuration(
    PreprocessingConfiguration,
    TrainingConfiguration,
    GeneralConfiguration,
    StaticConfiguration,
):

    def __init__(self):
        PreprocessingConfiguration.__init__(self)
        TrainingConfiguration.__init__(self)
        GeneralConfiguration.__init__(self)
        StaticConfiguration.__init__(self)

    def load_grid_search_files(self):
        if self.grid_search_config_file is None or self.grid_search_config_file == '':
            return

        with open(self.grid_search_configs_folder + self.grid_search_config_file, 'r') as f:
            self.grid_search_config = json.load(f)

    def load_config_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.prefixes = data['prefixes']

        self.float_features = data['float_features']
        self.integer_features = data['integer_features']
        self.categorical_features = data['categorical_features']
        self.primary_features = data['primary_features']
        self.secondary_features = data['secondary_features']
        self.categorical_nan_replacement = data['categorical_nan_replacement']

        self.features_used = sorted(list(set(self.primary_features).union(set(self.secondary_features))))

        self.label_renaming_mapping = data['label_renaming_mapping']
        self.one_hot_features_to_remove = data['one_hot_features_to_remove']
        self.transfer_from_train_to_test = data['transfer_from_train_to_test']
        self.unused_labels = data['unused_labels']
        self.labels_used = data['labels_used']
        self.rul_splits: dict = data['rul_splits']

        # Add np.inf as last entry such that the last interval is open end because some rul in config.json are > 100 ...
        for label in self.rul_splits.keys():
            split = self.rul_splits[label]
            split.append(np.inf)
            self.rul_splits[label] = split

        # Remove duplicates to ensure output is correct (would result in wrong sum of changed examples otherwise)
        self.unused_labels = list(set(self.unused_labels))
        self.transfer_from_train_to_test = list(set(self.transfer_from_train_to_test))

        self.kafka_webserver_topics: dict = data["kafka_webserver_topics"]
        self.kafka_sensor_topics: dict = data["kafka_sensor_topics"]
        self.kafka_txt_topics: dict = data["kafka_txt_topics"]
        self.kafka_failure_sim_topics: dict = data["kafka_failure_simulation_topics"]

        self.datasets_to_import_from_kafka = data['datasets_to_import_from_kafka']
        self.stored_datasets = data['stored_datasets']
        self.run_to_failure_info: dict = data['run_to_failure_info']
        self.aux_df_type_mapping: dict = data['aux_df_type_mapping']

    def change_current_model(self, model_name: str):
        self.filename_model_to_use = model_name if model_name.endswith('/') else model_name + '/'
        self.directory_model_to_use = self.models_folder + self.filename_model_to_use
