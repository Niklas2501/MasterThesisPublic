import os
import random
import time

import numpy as np
import tensorflow as tf

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameter
from execution.Evaluator import Evaluator
from stgcn.Dataset import Dataset
from stgcn.InputGenerator import InputGenerator
from stgcn.ModelBuilder import ModelBuilder
from stgcn.ModelTrainer import ModelTrainer


class STGCN:

    def __init__(self, config: Configuration, dataset: Dataset, training: bool, hyper: Hyperparameter = None):
        self.reset_state()

        self.config: Configuration = config
        self.dataset: Dataset = dataset
        self.training = training
        self.hyper: Hyperparameter = hyper
        self.model: tf.keras.Model = tf.keras.Model()
        self.trainer = None
        self.model_builder = None

        self.load_model()

    def load_model(self):

        # hyper object is not none in case of grid search -> Use the provided HPs.
        if self.hyper is None:

            if self.training:
                file_path = self.config.hyper_file
                print('Creating model based on {} hyperparameter file'.format(self.config.hyper_file), '\n')

            else:
                file_path = self.config.directory_model_to_use + 'hyperparameters_used.json'

            self.hyper = Hyperparameter()
            self.hyper.load_from_file(file_path, self.config.use_hyper_file)
            self.hyper.check_constrains()

        self.hyper.set_time_series_properties(self.dataset.time_series_length, self.dataset.time_series_depth)

        self.model_builder = ModelBuilder(self.config, self.hyper, self.dataset)

        if self.training:
            self.model = self.model_builder.get_compiled_model()
            self.trainer = ModelTrainer(self.config, self.hyper, self.dataset, self.model)
        else:
            self.restore_model()

    def restore_model(self):
        # Variant 1 - Full model is stored, would be preferred but not possible due to usage of graph layer.
        # self.model = tf.keras.models.load_model(self.config.directory_model_to_use + 'weights')

        # Variant 2 - Only weights are saved
        self.model = self.model_builder.get_uncompiled_model()
        self.model.load_weights(self.config.directory_model_to_use + 'weights')

        # Add softmax layer for directly interpretable output, assumes no activation was applied after the last layer.
        self.model = self.model_builder.get_prediction_model(self.model)

    def switch_to_testing(self):
        self.training = False

        # Add softmax layer for directly interpretable output, assumes no activation was applied in the models last layer.
        self.model = self.model_builder.get_prediction_model(self.model)

    def print_model_info(self):
        # tf.keras.utils.plot_model(self.model, "../logs/model.png", show_shapes=True, dpi=300)
        self.model.summary()
        print()

    def test_model(self, print_results=False, selected_model_name=None):

        if print_results:
            model_name = selected_model_name if selected_model_name is not None else self.config.filename_model_to_use
            print(f'Evaluating model {model_name} on the {self.config.part_tested} dataset. \n')

        evaluator = Evaluator(self.dataset, self.config.part_tested)
        input_generator = InputGenerator(self.config, self.hyper, self.dataset)

        test_data, tested_indices = input_generator.get_testing_data(test_failures_only=self.config.test_failures_only)

        start = time.perf_counter()
        predictions_softmax = self.model.predict(test_data)[0]
        predictions_class_index = np.argmax(predictions_softmax, axis=1)
        predictions_class_label = self.dataset.unique_labels_overall[predictions_class_index]
        evaluator.add_prediction_batch(tested_indices, predictions_class_label)
        end = time.perf_counter()
        evaluator.calculate_results()

        if print_results:
            evaluator.print_results(end - start)
            print('\nHyperparameter configuration used for this test:\n')
            self.hyper.print_hyperparameters()

        self.reset_state()

        return evaluator

    def train_model(self):
        return self.trainer.train_model()

    @staticmethod
    def reset_state():
        # Set seeds in order to get consistent results
        seeds = [25011997, 19011940, 15071936, 12101964, 24051997]
        seed = seeds[0]

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        np.random.seed(seed)
        random.seed(seed)

        # Reimporting tensorflow in necessary!
        import tensorflow as tf
        tf.random.set_seed(seed)
