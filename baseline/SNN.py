import numpy as np
import tensorflow as tf
from numpy.random import RandomState
from sklearn.base import BaseEstimator

from configuration.Configuration import Configuration
from stgcn.Dataset import Dataset
from stgcn.InputGenerator import InputGenerator
from stgcn.STGCN import STGCN


class SNN(BaseEstimator):

    def __init__(self, config: Configuration, dataset: Dataset, sim_norm=1, k=1, use_case_base=False, drop_dense=False):
        self.sim_norm = sim_norm
        self.k = k
        self.use_case_base = use_case_base
        self.drop_dense = drop_dense
        self.case_base_class_size = 150

        self.x_train_encoded = None
        self.y_train = None
        self.snn_sim_model = None

        self.stgcn = STGCN(config, dataset, training=False)
        self.input_generator = InputGenerator(config, self.stgcn.hyper, self.stgcn.dataset)
        self.encoder: tf.keras.Model = self.stgcn.model

    def fit(self, x_train, y_train):

        # Adapt the encoder model based on which layer out to be used
        last_layer_name = 'agg' if self.drop_dense else 'pred_out'
        output = self.encoder.get_layer(last_layer_name).get_output_at(0)

        if output.shape[1] > 1500:
            raise ValueError('Using feature vectors with such high dimensionality will likely result in OOM-Errors.')

        self.encoder = tf.keras.models.Model(self.encoder.inputs, output)

        if self.use_case_base:
            x_train, y_train = self.extract_case_base(x_train, y_train)

        # Note: predict is necessary, will otherwise result in error regarding creating a model from non symbolic tensors
        encoder_input = self.input_generator.get_dynamic_input(x_train)
        self.x_train_encoded = self.encoder.predict(encoder_input)
        self.snn_sim_model: tf.keras.Model = self.get_snn_sim_model(self.x_train_encoded)
        self.y_train = y_train

    def predict(self, x_test):
        if self.snn_sim_model is None:
            raise ValueError('fit methode must be called before predict can be used.')

        # Use the snn_sim_model to calculate the top k most similar examples in train for each example in test
        encoder_input = self.input_generator.get_dynamic_input(x_test)
        x_test_encoded = self.encoder.predict(encoder_input)
        snn_sim_model_output = self.snn_sim_model.predict(x_test_encoded)

        # Map the model output to the corresponding class labels and determine the most frequent class
        top_k_values, top_k_indices = snn_sim_model_output[0], snn_sim_model_output[1]
        top_k_classes = self.y_train[top_k_indices]
        y_pred = np.apply_along_axis(SNN.select_predicted_class, 1, top_k_classes)

        return y_pred

    # Note: Should be improved for k>1:
    # For equal number of classes in top k the class earliest in the alphabet is currently selected.
    # Improvement could be a decision based on number of examples in the dataset -> would strongly encourage no_failure
    # Alternative: distance based weighing
    @staticmethod
    def select_predicted_class(top_k_classes_example):
        u, count = np.unique(top_k_classes_example, return_counts=True)
        count_sort_ind = np.argsort(-count)

        # Sort the unique classes descendingly based on counts in the top k and return the most frequent one
        u = u[count_sort_ind]

        # Nasty bug: If using u_0 = u[0] numpy would infer the wrong dtype and possibly truncates the class string
        # Source with alternative fix: https://github.com/numpy/numpy/issues/8352#issuecomment-304462725
        u_0 = np.array(u[0], object)

        return u_0

    def get_snn_sim_model(self, x_train_encoded):
        x_train_encoded = tf.constant(value=x_train_encoded)

        snn_sim_input = tf.keras.Input(shape=x_train_encoded.shape[1:])
        x = snn_sim_input

        # Add a dimension such that each input example is compared to all examples in x_train_encoded
        # via broadcasting
        x = tf.expand_dims(x, axis=1)

        # Calculate the distances between each input example and all examples in x_train_encoded
        # Multiply distance by -1 such that top k selects the ones with the smallest distance
        distance = tf.subtract(x, x_train_encoded, name='distance')
        distance = tf.norm(distance, ord=self.sim_norm, axis=2, name='norm')
        distance_inv = tf.negative(distance)

        # Get indices and distance values for the k most similar examples in x_train_encoded
        # Multiply the distance values by -1 again to get the initial ones
        top_k_values, top_k_indices = tf.math.top_k(distance_inv, self.k, name='knn_selection')
        top_k_values = tf.negative(top_k_values)

        return tf.keras.Model(snn_sim_input, [top_k_values, top_k_indices], name='snn_sim_model')

    def extract_case_base(self, x_train, y_train):

        # Get unique classes
        classes = np.unique(y_train)

        # For each class get the indices of all examples with this class
        indices_of_classes = []
        for c in classes:
            indices_of_classes.append(np.where(y_train == c)[0])

        # Reduce classes to equal many examples
        new_indices = []
        random = np.random.default_rng(seed=self.stgcn.config.random_seed)

        for indices in indices_of_classes:
            # If there are less examples than there should be for each class only those can be used
            epc = self.case_base_class_size if self.case_base_class_size < len(indices) else len(indices)
            new_indices.extend(random.choice(indices, epc, replace=False))

        x_train = x_train[new_indices, :, :]
        y_train = y_train[new_indices]

        return x_train, y_train
