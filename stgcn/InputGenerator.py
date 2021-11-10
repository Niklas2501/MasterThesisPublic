import numpy as np
import pandas as pd
import tensorflow as tf

from configuration.Configuration import Configuration
from configuration.Enums import *
from configuration.Hyperparameter import Hyperparameter
from stgcn.Dataset import Dataset


class InputGenerator:

    def __init__(self, config: Configuration, hyper: Hyperparameter, dataset: Dataset):
        self.config = config
        self.dataset = dataset
        self.hyper = hyper

    def load_a_input(self):

        if A_Variant.uses_predefined(self.hyper.a_variant):
            path = self.config.get_additional_data_path(self.config.a_pre_file)
            a_df = pd.read_excel(path, engine='openpyxl')
            a_df = a_df.set_index('Features')

            col_values = a_df.columns.values
            index_values = a_df.index.values

            if not np.array_equal(col_values, self.dataset.get_feature_names()):
                raise ValueError('Ordering of features in the adjacency matrix (columns) '
                                 'does not match the one in the dataset.')

            if not np.array_equal(index_values, self.dataset.get_feature_names()):
                raise ValueError('Ordering of features in the adjacency matrix (index) '
                                 'does not match the one in the dataset.')

            a_input = a_df.values.astype(dtype=np.float)

        elif A_Variant.is_emb_variant(self.hyper.a_variant):
            path = self.config.get_additional_data_path(self.config.embeddings_file)
            a_input = np.load(path)
        elif self.hyper.a_variant == A_Variant.A_I:
            a_input = np.eye(len(self.dataset.feature_names))
        elif self.hyper.a_variant == A_Variant.A_DEV:
            a_input = np.ones(shape=(len(self.dataset.feature_names), len(self.dataset.feature_names)))
        else:
            raise ValueError('Undefined adjacency matrix variant:', self.hyper.a_variant)

        if Modification.FORCE_IDENTITY in self.hyper.modifications and A_Variant.uses_predefined(self.hyper.a_variant):
            np.fill_diagonal(a_input, self.hyper.loop_weight, wrap=False)

        return a_input

    @staticmethod
    def check_symmetry(a, tol=1e-8):
        return np.all(np.abs(a - a.T) < tol)

    @staticmethod
    def graph_preprocessing(a, hyper, outside_of_graph):

        # If part of the gsl component the learned adjacency matrix is always assumed to not be symmetric.
        is_sym = InputGenerator.check_symmetry(a) if outside_of_graph else False

        if hyper.gnn_type == GNN_Type.GAT and outside_of_graph:
            # Just check that a is a binary matrix
            if not ((a == 0) | (a == 1)).all():
                raise ValueError('A must be binary when using the GAT layer.')
        elif hyper.gnn_type == GNN_Type.GCN:
            a = InputGenerator.gcn_preprocessing(a, is_sym, outside_of_graph)
        elif hyper.gnn_type == GNN_Type.DIFF:
            a = InputGenerator.diff_preprocessing(a, is_sym, outside_of_graph)
        elif hyper.gnn_type == GNN_Type.ARMA:
            a = InputGenerator.gcn_preprocessing(a, is_sym, outside_of_graph, add_I=True)
        return a

    @staticmethod
    def diff_preprocessing(A: tf.Tensor, symmetric, outside_of_graph):
        # Corrections necessary for serious usage
        # Spektral layer can only deal with one direction diffusion process, i.e. a symmetric adj
        symmetric = True

        D_out_values = tf.math.reduce_sum(A, axis=1)
        D_out_pow = tf.linalg.inv(tf.linalg.diag(D_out_values, name='D'))
        T_f = tf.linalg.matmul(D_out_pow, A)

        if not symmetric:
            D_in_values = tf.math.reduce_sum(A, axis=2)
            D_in_pow = tf.linalg.inv(tf.linalg.diag(D_in_values, name='D'))
            T_b = tf.linalg.matmul(a=D_in_pow, b=A, transpose_b=True)
            out = [T_f, T_b]
        else:
            out = [T_f]

        if outside_of_graph:
            return [part.numpy() for part in out][0]  # Remove index selection
        else:
            return out

    @staticmethod
    def gcn_preprocessing(A: tf.Tensor, symmetric, outside_of_graph=False, add_I=True):

        # ArmaConv uses the same preprocessing just without the added self loops.
        if add_I:
            A = tf.add(A, tf.linalg.diag(tf.ones(shape=(1, A.shape[1]), dtype='float32'), name='I'), 'add_I')
        else:
            A = tf.expand_dims(A, axis=0)

        D_diag_values = tf.math.reduce_sum(A, axis=2, name='row_sum')
        D = tf.linalg.diag(D_diag_values, name='D')

        if symmetric:
            # https://de.wikipedia.org/wiki/Matrixpotenz#Negative_Exponenten
            # https://de.wikipedia.org/wiki/Quadratwurzel_einer_Matrix#Definition
            D_pow = tf.linalg.sqrtm(tf.linalg.inv(D))
            A_hat = tf.linalg.matmul(tf.linalg.matmul(D_pow, A), D_pow, name='A_hat')
        else:
            # https://github.com/tkipf/gcn/issues/91#issuecomment-469181790
            D_pow = tf.linalg.inv(D)
            A_hat = tf.linalg.matmul(D_pow, A, name='A_hat')

        if outside_of_graph:
            # Convert from tensor to numpy array and remove artificial batch dimension.
            return np.squeeze(A_hat.numpy(), axis=0)
        else:
            return A_hat

    @staticmethod
    def repeat_a_input(a, repetitions):
        x = a.copy()
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, repetitions, axis=0)
        return x

    def get_training_data(self):
        x_train, y_train = self.dataset.get_train().get_x(), self.dataset.get_train().get_y()
        x_val, y_val = self.dataset.get_train_val().get_x(), self.dataset.get_train_val().get_y()

        # Add adjacency matrix input in addition to main time series data
        a_input = self.load_a_input()

        if A_Variant.uses_predefined(self.hyper.a_variant):
            a_input = InputGenerator.graph_preprocessing(a_input, self.hyper, outside_of_graph=True)

        a_input_train = self.repeat_a_input(a_input, x_train.shape[0])
        a_input_val = self.repeat_a_input(a_input, x_val.shape[0])

        train_input = (x_train, a_input_train)
        val_data = ((x_val, a_input_val), y_val)

        return train_input, y_train, val_data

    def get_testing_data(self, test_failures_only):
        part = self.dataset.get_part(self.config.part_tested)
        x_tested = part.get_x()

        if test_failures_only:
            indices = part.get_failure_indices()
            x_tested = x_tested[indices]
        else:
            indices = np.arange(x_tested.shape[0])

        a_input = self.load_a_input()

        if A_Variant.uses_predefined(self.hyper.a_variant):
            a_input = InputGenerator.graph_preprocessing(a_input, self.hyper, outside_of_graph=True)

        a_input = self.repeat_a_input(a_input, x_tested.shape[0])

        return (x_tested, a_input), indices

    def get_dynamic_input(self, main_input):
        a_input = self.load_a_input()

        if A_Variant.uses_predefined(self.hyper.a_variant):
            a_input = InputGenerator.graph_preprocessing(a_input, self.hyper, outside_of_graph=True)

        a_input = self.repeat_a_input(a_input, main_input.shape[0])

        return main_input, a_input
