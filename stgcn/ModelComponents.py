import numpy as np
import spektral
import tensorflow as tf
from tensorflow.keras import layers

from configuration.Configuration import Configuration
from configuration.Enums import *
from configuration.Hyperparameter import Hyperparameter
from stgcn.InputGenerator import InputGenerator


# noinspection DuplicatedCode
def st_block_1d(x, a, hyper: Hyperparameter, block_id, cnn_params, graph_model=None):
    pre_graph_cnn, post_graph_cnn = np.split(cnn_params, 2, axis=0) if hyper.conv_after_graph else [cnn_params, []]
    ts_depth = x.shape[2]

    block_input = x

    for i, layer in enumerate(pre_graph_cnn):
        f, s, d = tuple(layer)

        x = layers.Conv1D(filters=f, kernel_size=(s,), dilation_rate=(d,),
                          padding=hyper.cnn_1D_padding, name=f'temporal_cnn_{block_id}_{i}')(x)
        x = layers.BatchNormalization(scale=False, name=f'bn_{block_id}_{i}')(x)
        x = get_activation_instance(hyper.activation, name=f'act_{block_id}_{i}')(x)

    # Aggregate to initial number of features using 1x1 convolutions
    if x.shape[-1] != ts_depth:
        x = layers.Conv1D(filters=ts_depth, kernel_size=(1,), dilation_rate=1,
                          padding=hyper.cnn_1D_padding, name=f'pre_graph_1x1_{block_id}')(x)
        x = layers.BatchNormalization(scale=False, name=f'pre_graph_bn_{block_id}')(x)
        x = get_activation_instance(hyper.activation, name=f'pre_graph_act_{block_id}')(x)

    x, residual_source = res_connection(hyper, x=x, block_input=block_input, block_id=block_id)
    x = layers.SpatialDropout1D(rate=hyper.dropout_rate, name=f'spatial_dropout_{block_id}')(x)
    x, a_out, graph_model = graph_layer(x, a, hyper, block_id, graph_model=graph_model)

    nbr_pre_cnn = len(pre_graph_cnn)
    if len(post_graph_cnn) > 0:
        x = get_activation_instance(hyper.activation)(x)

        for i, layer in enumerate(post_graph_cnn):
            f, s, d = tuple(layer)

            x = layers.Conv1D(filters=f, kernel_size=(s,), dilation_rate=(d,),
                              padding=hyper.cnn_1D_padding, name=f'temporal_cnn_{block_id}_{nbr_pre_cnn + i}')(x)
            x = layers.BatchNormalization(scale=False, name=f'bn_{block_id}_{nbr_pre_cnn + i}')(x)
            x = get_activation_instance(hyper.activation, name=f'act_{block_id}_{nbr_pre_cnn + i}')(x)

        if x.shape[-1] != ts_depth:
            x = layers.Conv1D(filters=ts_depth, kernel_size=(1,), dilation_rate=1,
                              padding=hyper.cnn_1D_padding, name=f'post_graph_1x1_{block_id}')(x)
            x = layers.BatchNormalization(scale=False, name=f'post_graph_bn_{block_id}')(x)
            x = get_activation_instance(hyper.activation, name=f'post_graph_act_{block_id}')(x)

    # Connect residual flow
    x = tf.add(x, residual_source, name=f'res_con_{block_id}')
    x = get_activation_instance(hyper.activation, name=f'final_act_{block_id}')(x)

    return x, a_out, graph_model


# noinspection DuplicatedCode
def st_block_1d_grouped(x, a, hyper: Hyperparameter, block_id, cnn_params, graph_model=None):
    pre_graph_cnn, post_graph_cnn = np.split(cnn_params, 2, axis=0) if hyper.conv_after_graph else [cnn_params, []]
    nodes = x.shape[2]

    block_input = x

    for i, layer in enumerate(pre_graph_cnn):
        filters_per_node, s, d = tuple(layer)

        x = layers.Conv1D(filters=filters_per_node * nodes, groups=nodes, kernel_size=(s,), dilation_rate=(d,),
                          padding=hyper.cnn_1D_padding, name=f'temporal_cnn_{block_id}_{i}')(x)

        x = layers.BatchNormalization(scale=False, name=f'bn_{block_id}_{i}')(x)
        x = get_activation_instance(hyper.activation, name=f'act_{block_id}_{i}')(x)

        if Modification.GROUPED_1D_1x1_AGG in hyper.modifications and x.shape[-1] != nodes:
            x = layers.Conv1D(filters=nodes, kernel_size=(1,), dilation_rate=1,
                              padding=hyper.cnn_1D_padding, name=f'1d_1x1_{block_id}_{i}')(x)
        if Modification.GROUPED_G_1x1_AGG in hyper.modifications and x.shape[-1] != nodes:
            # x = layers.Reshape(target_shape=(x.shape[1], nodes, filters_per_node), name=f'2d_reshape_{block_id}_{i}')(x)
            # x = layers.Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=(1, 1), padding='same',
            #                   name=f'2d_1x1_{block_id}_{i}')(x)
            # x = tf.squeeze(x, axis=-1, name=f'2d_squeeze_{block_id}_{i}')
            x = layers.Conv1D(filters=1 * nodes, groups=nodes, kernel_size=(1,), dilation_rate=1,
                              padding=hyper.cnn_1D_padding, name=f'g_1x1_{block_id}_{i}')(x)
        elif x.shape[-1] != nodes:
            raise ValueError('No filter aggregation method for 1d selected.')

        x = layers.BatchNormalization(scale=False, name=f'1x1_bn_{block_id}_{i}')(x)
        x = get_activation_instance(hyper.activation, name=f'1x1_act_{block_id}_{i}')(x)

    x, residual_source = res_connection(hyper, x=x, block_input=block_input, block_id=block_id)
    x = layers.SpatialDropout1D(rate=hyper.dropout_rate, name=f'pre_graph_dropout_{block_id}')(x)
    x, a_out, graph_model = graph_layer(x, a, hyper, block_id, graph_model=graph_model)

    # Repeat temporal convolution after graph layer if enabled.
    nbr_pre_cnn = len(pre_graph_cnn)
    if len(post_graph_cnn) > 0:
        x = get_activation_instance(hyper.activation)(x)

        for i, layer in enumerate(post_graph_cnn):
            filters_per_node, s, d = tuple(layer)

            x = layers.Conv1D(filters=filters_per_node * nodes, groups=nodes, kernel_size=(s,), dilation_rate=(d,),
                              padding=hyper.cnn_1D_padding, name=f'temporal_cnn_{block_id}_{i + nbr_pre_cnn}')(x)

            x = layers.BatchNormalization(scale=False, name=f'bn_{block_id}_{i + nbr_pre_cnn}')(x)
            x = get_activation_instance(hyper.activation, name=f'act_{block_id}_{i + nbr_pre_cnn}')(x)

            if Modification.GROUPED_1D_1x1_AGG in hyper.modifications and x.shape[-1] != nodes:
                x = layers.Conv1D(filters=nodes, kernel_size=(1,), dilation_rate=1,
                                  padding=hyper.cnn_1D_padding, name=f'1d_1x1_{block_id}_{i + nbr_pre_cnn}')(x)
            if Modification.GROUPED_G_1x1_AGG in hyper.modifications and x.shape[-1] != nodes:
                # x = layers.Reshape(target_shape=(x.shape[1], nodes, filters_per_node), name=f'1x1_reshape_{block_id}_{i+nbr_pre_cnn}')(x)
                # x = layers.Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=(1, 1), padding='same',
                #                   name=f'g_1x1_{block_id}_{i+nbr_pre_cnn}')(x)
                # x = tf.squeeze(x, axis=-1, name=f'1x1_squeeze_{block_id}_{i+nbr_pre_cnn}')
                x = layers.Conv1D(filters=1 * nodes, groups=nodes, kernel_size=(1,), dilation_rate=1,
                                  padding=hyper.cnn_1D_padding, name=f'g_1x1_{block_id}_{i + nbr_pre_cnn}')(x)
            elif x.shape[-1] != nodes:
                raise ValueError('No filter aggregation method for 1d selected.')

            x = layers.BatchNormalization(scale=False, name=f'1x1_bn_{block_id}_{i + nbr_pre_cnn}')(x)
            x = get_activation_instance(hyper.activation, name=f'1x1_act_{block_id}_{i + nbr_pre_cnn}')(x)

    # Connect residual flow
    x = tf.add(x, residual_source, name=f'res_con_{block_id}')
    x = get_activation_instance(hyper.activation, name=f'final_act_{block_id}')(x)

    return x, a_out, graph_model


# noinspection DuplicatedCode
def st_block_2d(x, a, hyper: Hyperparameter, block_id, cnn_params, graph_model=None):
    pre_graph_cnn, post_graph_cnn = np.split(cnn_params, 2, axis=0) if hyper.conv_after_graph else [cnn_params, []]

    block_input = x

    # Conv2D expects 4D input, add dimension for each feature
    x = tf.expand_dims(x, axis=-1, name=f'pre_graph_expand_{block_id}')

    for i, layer in enumerate(pre_graph_cnn):
        f, s, d = tuple(layer)

        x = layers.Conv2D(filters=f, kernel_size=(s, 1), dilation_rate=(d, 1), padding='same',
                          name=f'temporal_cnn_{block_id}_{i}')(x)
        x = layers.BatchNormalization(scale=False, name=f'bn_{block_id}_{i}')(x)
        x = get_activation_instance(hyper.activation, name=f'act_{block_id}_{i}')(x)

    # Aggregate to initial number of features
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=(1, 1),
                      padding='same', name=f'pre_graph_1x1_{block_id}')(x)
    x = layers.BatchNormalization(scale=False, name=f'pre_graph_bn_{block_id}')(x)
    x = get_activation_instance(hyper.activation, name=f'pre_graph_act_{block_id}')(x)

    # Remove the additional dimension again such it can be used as graph layer input
    x = tf.squeeze(x, axis=-1, name=f'pre_graph_squeeze_{block_id}')

    x, residual_source = res_connection(hyper, x=x, block_input=block_input, block_id=block_id)
    x = layers.SpatialDropout1D(rate=hyper.dropout_rate, name=f'pre_graph_dropout_{block_id}')(x)
    x, a_out, graph_model = graph_layer(x, a, hyper, block_id, graph_model=graph_model)

    # Repeat temporal convolution after graph layer if enabled.
    nbr_pre_cnn = len(pre_graph_cnn)
    if len(post_graph_cnn) > 0:
        x = get_activation_instance(hyper.activation)(x)

        x = tf.expand_dims(x, axis=-1, name=f'post_graph_expand_{block_id}')

        for i, layer in enumerate(post_graph_cnn):
            f, s, d = tuple(layer)

            x = layers.Conv2D(filters=f, kernel_size=(s, 1), dilation_rate=(d, 1),
                              padding='same', name=f'temporal_cnn_{block_id}_{nbr_pre_cnn + i}')(x)
            x = layers.BatchNormalization(scale=False, name=f'bn_{block_id}_{nbr_pre_cnn + i}')(x)
            x = get_activation_instance(hyper.activation, name=f'act_{block_id}_{nbr_pre_cnn + i}')(x)

        x = layers.Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=(1, 1),
                          padding='same', name=f'post_graph_1x1_{block_id}')(x)
        x = layers.BatchNormalization(scale=False, name=f'post_graph_bn_{block_id}')(x)
        x = get_activation_instance(hyper.activation, name=f'post_graph_act_{block_id}')(x)

        x = tf.squeeze(x, axis=-1, name=f'post_graph_squeeze_{block_id}')

    # Connect residual flow
    x = tf.add(x, residual_source, name=f'res_con_{block_id}')
    x = get_activation_instance(hyper.activation, name=f'final_act_{block_id}')(x)

    return x, a_out, graph_model


def res_connection(hyper, x, block_input, block_id):
    if hyper.res_connection == ResConnection.V0:  # V0 in thesis
        residual_source = block_input
    elif hyper.res_connection == ResConnection.V1:
        residual_source = x
    elif hyper.res_connection == ResConnection.V2:
        residual_source = x
        x = tf.add(residual_source, block_input, name=f'res_con_v2_{block_id}')
    elif hyper.res_connection == ResConnection.V3:  # V2 in thesis
        x = tf.add(x, block_input, name=f'res_con_v3_{block_id}')
        residual_source = x
    elif hyper.res_connection == ResConnection.V4:  # V1 in thesis
        residual_source = tf.add(block_input, x, name=f'res_con_v4_{block_id}')
    elif hyper.res_connection == ResConnection.SEQ:
        # In case a sequential model is created no residual connection should be added here because multiple
        # graph layers could be added. Note that this connection will still appear in the model structure.
        residual_source = 0.0
    else:
        raise ValueError('ResConnection variant unknown.')

    return x, residual_source


def graph_layer(x, a, hyper: Hyperparameter, block_id, graph_model=None):
    # No graph model passed means that individual (non-shared) weights are used (or this is the first st-block).
    if graph_model is None:

        if hyper.gnn_type == GNN_Type.GAT:
            graph_model = spektral.layers.GATConv(channels=hyper.time_series_length,
                                                  attn_heads=hyper.gnn_layer_parameter,
                                                  return_attn_coef=True, concat_heads=False)
        elif hyper.gnn_type == GNN_Type.GCN:
            graph_model = spektral.layers.GCNConv(channels=hyper.time_series_length)
        elif hyper.gnn_type == GNN_Type.DIFF:
            graph_model = spektral.layers.DiffusionConv(channels=hyper.time_series_length, K=hyper.gnn_layer_parameter)
        elif hyper.gnn_type == GNN_Type.ARMA:
            graph_model = spektral.layers.ARMAConv(channels=hyper.time_series_length, order=1,
                                                   iterations=hyper.gnn_layer_parameter)
        else:
            raise NotImplementedError('GNN_Type not implemented.')

    # Graph layer expects nodes = features in the first dimension, time steps = node features in the second one.
    x = layers.Permute(dims=(2, 1), name=f'pre_graph_perm_{block_id}')(x)

    graph_out = graph_model([x, a])

    # If the GAT layer is used, replace the A input with the returned adjacency matrix.
    x, a_out = graph_out if hyper.gnn_type == GNN_Type.GAT else (graph_out, a)

    # Undo the switch of dimensions.
    x = layers.Permute(dims=(2, 1), name=f'post_graph_perm_{block_id}')(x)

    # Note: Activation should not be done here in case a residual connection is added after the graph layer
    # after which the activation should be done.
    # x = layers.ReLU()(x)

    return x, a_out, graph_model


class AParamWrapper(tf.keras.layers.Layer):
    def __init__(self, a_input, a_variant, random_init, **kwargs):
        super().__init__(**kwargs)
        self.a_variant = a_variant

        if self.a_variant == A_Variant.A_PARAM:

            if random_init:
                initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.25)
            else:
                initializer = tf.keras.initializers.constant(a_input)

            self.a_params = self.add_weight(shape=(1, a_input.shape[1], a_input.shape[2]), initializer=initializer,
                                            trainable=True, name='a_params')
        elif self.a_variant == A_Variant.A_OPT:
            self.a_params = self.add_weight(shape=(1, a_input.shape[1], a_input.shape[2]),
                                            initializer=tf.keras.initializers.Zeros(),
                                            trainable=True, name='a_params')
        elif A_Variant.is_emb_variant(self.a_variant):
            trainable = self.a_variant == A_Variant.A_EMB_V2

            if random_init:
                initializer = tf.keras.initializers.RandomNormal
                e1_init, e2_init = initializer(mean=0.5, stddev=0.25), initializer(mean=0.5, stddev=0.25)
            else:
                e1_init, e2_init = tf.keras.initializers.constant(a_input), tf.keras.initializers.constant(a_input)

            self.e1 = self.add_weight(shape=(1, a_input.shape[0], a_input.shape[1]), trainable=trainable, name='e1',
                                      initializer=e1_init)
            self.e2 = self.add_weight(shape=(1, a_input.shape[0], a_input.shape[1]), trainable=trainable, name='e2',
                                      initializer=e2_init)

        else:
            raise ValueError('Undef. A_Variant:', self.a_variant)

    def call(self, inputs, **kwargs):
        if self.a_variant in [A_Variant.A_PARAM, A_Variant.A_OPT]:
            return self.a_params
        elif A_Variant.is_emb_variant(self.a_variant):
            return self.e1, self.e2
        else:
            raise ValueError()

    def get_config(self):
        config = super(AParamWrapper, self).get_config()

        if A_Variant.is_emb_variant(self.a_variant):
            config.update({"e1": self.e1})
            config.update({"e2": self.e2})
        else:
            config.update({"a_params": self.a_params})

        return config


def gsl_block(a, hyper: Hyperparameter, config: Configuration, dataset: Dataset):
    ar = tf.keras.regularizers.L1() if Modification.L1_REG in hyper.modifications else None
    random_init = Modification.GSL_RANDOM_INIT in hyper.modifications

    if hyper.a_variant in [A_Variant.A_PRE, A_Variant.A_I, A_Variant.A_DEV]:
        pass
    elif hyper.a_variant == A_Variant.A_OPT:

        # Based on Fatemi et al. 2021 https://arxiv.org/abs/2102.05034, apart from the activity_regularizer
        a_l = AParamWrapper(a, a_variant=hyper.a_variant, random_init=random_init, name='a_l')(a)
        a_l = layers.Layer(name='a_l_passthrough')(a_l)
        a = tf.add(a, a_l, name='a_opt_res')
        # a = layers.Layer(name='a_l_passthrough_2', activity_regularizer=ar)(a)
        # a = layers.LeakyReLU(alpha=0.1, name='a_opt_leaky_act', activity_regularizer=ar)(a)
        # a = layers.ELU(name='a_opt_act', activity_regularizer=ar)(a)
        # a = tf.add(a, 1, name='a_opt_add')

    elif hyper.a_variant == A_Variant.A_PARAM:
        # Note: Don't use a for initialisation --> Net input, changes
        ig = InputGenerator(config, hyper, dataset)
        a_pred = ig.load_a_input()
        a_pred = a_pred[np.newaxis, :, :]

        a = AParamWrapper(a_pred, a_variant=hyper.a_variant, random_init=random_init, name='a_param')(a)
        a = layers.Layer(name='a_l_passthrough', activity_regularizer=ar)(a)
        # a = layers.LeakyReLU(alpha=0.1, name='a_param_leaky_act', activity_regularizer=ar)(a)
        # a = layers.ELU(name='a_param_act', activity_regularizer=ar)(a)
        # a = tf.add(a, 1, name='a_param_add')

    elif A_Variant.is_emb_variant(hyper.a_variant):

        ig = InputGenerator(config, hyper, dataset)
        e_init = ig.load_a_input()

        # a_emb variant based on Directed-A by Wu et al. (https://arxiv.org/abs/2005.11650),
        # apart from the k-NN selection being omitted in favor of a L1 regularisation of the resulting a
        if hyper.a_variant == A_Variant.A_EMB_V1:

            e1, e2 = AParamWrapper(e_init, a_variant=hyper.a_variant, random_init=random_init, name='a_emb')(a)
            e1 = layers.Dense(hyper.a_emb_dense_dim, activation=None, name='e1_lin_transform')(e1)
            e2 = layers.Dense(hyper.a_emb_dense_dim, activation=None, name='e2_lin_transform')(e2)

            m1 = tf.keras.layers.Activation('tanh', name='e1_tanh')(hyper.a_emb_alpha * e1)
            m2 = tf.keras.layers.Activation('tanh', name='e2_tanh')(hyper.a_emb_alpha * e2)

            a = tf.matmul(m1, m2, transpose_b=True, name='a_emb_mat_mul')
            a = tf.keras.layers.Activation('tanh', name='a_emb_tanh')(hyper.a_emb_alpha * a)
            a = layers.ReLU(name='a_emb_relu', activity_regularizer=ar)(a)

        elif hyper.a_variant == A_Variant.A_EMB_V2:

            # Adapted a_emb variant that uses the node embeddings as initialisation of parameters
            # and trains those directly. Uses
            e1, e2 = AParamWrapper(e_init, a_variant=hyper.a_variant, random_init=random_init, name='a_emb')(a)

            # Does nothing, but necessary for model construction
            e1 = layers.Layer(name='e1_passthrough')(e1)
            e2 = layers.Layer(name='e2_passthrough')(e2)

            # Wu et al.s activation method (https://arxiv.org/abs/1906.00121)
            a = tf.matmul(e1, e2, transpose_b=True, name='a_emb_mat_mul')
            a = layers.ReLU(name='a_emb_act', activity_regularizer=ar)(a)
            a = layers.Softmax(name='a_emb_softmax')(a)

        elif hyper.a_variant == A_Variant.A_EMB_V3:

            e1, e2 = AParamWrapper(e_init, a_variant=hyper.a_variant, random_init=random_init, name='a_emb')(a)
            e1 = layers.Dense(hyper.a_emb_dense_dim, name='e1_lin_transform')(e1)
            e2 = layers.Dense(hyper.a_emb_dense_dim, name='e2_lin_transform')(e2)

            e1 = layers.BatchNormalization(scale=False, name='e1_bn')(e1)
            e2 = layers.BatchNormalization(scale=False, name='e2_bn')(e2)
            e1 = layers.LeakyReLU(alpha=0.1, name='e1_leaky')(e1)
            e2 = layers.LeakyReLU(alpha=0.1, name='e2_leaky')(e2)

            a = tf.matmul(e1, e2, transpose_b=True, name='a_emb_mat_mul')
            a = layers.LeakyReLU(alpha=0.1, name='a_emb_act', activity_regularizer=ar)(a)
            a = layers.Softmax(name='a_emb_softmax')(a)

    else:
        raise ValueError('Unknown A variant:', hyper.a_variant)

    if not A_Variant.uses_predefined(hyper.a_variant) and Modification.FORCE_IDENTITY in hyper.modifications:
        I = tf.constant(shape=(1, a.shape[1]), value=hyper.loop_weight, dtype='float32')
        a = tf.linalg.set_diag(a, diagonal=I, name='add_I')

    # Apply preprocessing to learned adjacency matrix if enabled
    if Modification.FORCE_NORMALISATION in hyper.modifications and hyper.a_variant != A_Variant.A_PRE:
        a = InputGenerator.graph_preprocessing(a, hyper, False)

    # GAT layer uses a binary adj, therefore use a knn approach for the conversion in case of a learned adj
    if hyper.gnn_type == GNN_Type.GAT and hyper.a_variant not in [A_Variant.A_PRE, A_Variant.A_I, A_Variant.A_DEV]:
        a = knn_reduction(a, hyper.var_parameter, convert_to_binary=True)
    # For other types the knn approach might also be used to induce a sparse matrix as an alternative to the L1_reg
    elif Modification.KNN_RED in hyper.modifications:
        a = knn_reduction(a, hyper.var_parameter, convert_to_binary=False)

    return a


def knn_reduction(a, k, convert_to_binary=False):
    # Transpose because neighbourhood is defined column wise but top_k is calculated per row.
    a = tf.transpose(a, perm=[0, 2, 1])

    # Get the kth largest value per row, transpose in necessary such that a row wise comparison is done in tf.where.
    top_k = tf.math.top_k(a, k).values
    kth_largest = tf.reduce_min(top_k, axis=-1, keepdims=True)

    # If convert_to_binary a connection is added for the k nearest neighbours,
    # otherwise the weight values of those are kept and only the ones below the threshold are set to 0.
    value = 1.0 if convert_to_binary else a
    a = tf.where(a < kth_largest, 0.0, value)

    # Reverse initial transpose.
    a = tf.transpose(a, perm=[0, 2, 1])

    return a


def output_block(x, hyper: Hyperparameter, num_classes, alt_out):
    if hyper.permute_before_agg:
        x = layers.Permute(dims=[2, 1], name='pre_agg_perm')(x)

    if hyper.final_agg == FinalAggregation.GLOBAL_AVG:
        x = spektral.layers.GlobalAvgPool(name='agg')(x)
    elif hyper.final_agg == FinalAggregation.GLOBAL_MAX:
        x = spektral.layers.GlobalMaxPool(name='agg')(x)
    elif hyper.final_agg == FinalAggregation.GLOBAL_SUM:
        x = spektral.layers.GlobalSumPool(name='agg')(x)
    elif hyper.final_agg == FinalAggregation.GLOBAL_ATT:
        x = spektral.layers.GlobalAttentionPool(channels=hyper.var_parameter, name='agg')(x)
    elif hyper.final_agg == FinalAggregation.GLOBAL_ATT_SUM:
        x = spektral.layers.GlobalAttnSumPool(name='agg')(x)
    elif hyper.final_agg == FinalAggregation.CUSTOM_FLATTEN:

        # Note: Only behaves as intended if input has shape (batch, features, time steps)
        # i.e. permute_before_agg == True (xor direct output of a graph layer, not the block resulting from graph_layer)
        x = layers.Reshape(target_shape=(x.shape[1] * x.shape[2],), name='agg')(x)

    elif hyper.final_agg == FinalAggregation.RNN_AGG:

        # Note: If not permute_before_agg will aggregate features along time dimension
        # else will aggregate features per time step
        x = layers.GRU(units=x.shape[-1], return_sequences=False)(x)
        x = get_activation_instance(hyper.activation, name='agg')(x)

    alt_out = x

    # Note that there is no (softmax) activation function. See: https://www.tensorflow.org/tutorials/quickstart/beginner
    pred_out = layers.Dense(num_classes, name="pred_out")(x)

    return pred_out, alt_out


def get_activation_instance(activation, name=None):
    if activation == Activation.RELU:
        return layers.ReLU(name=name)
    elif activation == Activation.LEAKY_RELU:
        return layers.LeakyReLU(alpha=0.1, name=name)
    elif activation == Activation.ELU:
        return layers.ELU(alpha=1.0, name=name)


class WCCE(tf.keras.losses.Loss):

    def __init__(self, weights):
        super().__init__()

        # Note that the [] around weights is intentional to get the right shape
        self.weights = tf.constant([weights], dtype='float32')

    # Based on https://stackoverflow.com/questions/44560549/unbalanced-data-and-weighted-cross-entropy
    def call(self, y_true, y_pred):
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(self.weights * y_true, axis=1)

        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)

        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights

        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)

        return loss
