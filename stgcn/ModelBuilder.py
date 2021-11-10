from tensorflow.keras.metrics import *

from stgcn.ModelComponents import *


class ModelBuilder:

    def __init__(self, config: Configuration, hyper: Hyperparameter, dataset: Dataset):
        self.config = config
        self.hyper = hyper
        self.dataset = dataset

        self.a_input_shape = (self.hyper.time_series_depth, self.hyper.time_series_depth)
        self.ts_input_shape = (self.hyper.time_series_length, self.hyper.time_series_depth)

    def get_compiled_model(self):
        model = self.get_uncompiled_model()

        if self.hyper.gradient_cap_enabled:
            optimizer = tf.optimizers.Adam(self.hyper.learning_rate, clipnorm=self.hyper.gradient_cap)
        else:
            optimizer = tf.optimizers.Adam(self.hyper.learning_rate)

        if self.hyper.loss_function == LossFunction.CATEGORICAL_CROSSENTROPY:
            # from_logits = True: https://www.tensorflow.org/tutorials/quickstart/beginner
            main_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        elif LossFunction.is_wcce_variant(self.hyper.loss_function):
            main_loss = WCCE(self.dataset.get_class_weights(self.hyper.loss_function))
        else:
            raise ValueError('Loss function not implemented.')

        model.compile(
            optimizer=optimizer,
            loss={"pred_out": main_loss},
            metrics={"pred_out": [CategoricalAccuracy()]}
        )

        return model

    def get_uncompiled_model(self):

        if ArchitectureVariant.block_based_architecture_selected(self.hyper.architecture_variant):
            return self.get_uncompiled_stgcn_model()
        elif self.hyper.architecture_variant == ArchitectureVariant.SEQUENTIAL:
            return self.get_uncompiled_seq_model()
        elif self.hyper.architecture_variant == ArchitectureVariant.DEV:
            return self.get_uncompiled_dev_model()

    def get_uncompiled_stgcn_model(self):

        #
        # Graph structure learning
        #

        a_input = layers.Input(self.a_input_shape, name='a_input', sparse=False)
        a = gsl_block(a_input, self.hyper, self.config, self.dataset)
        a_out = a

        #
        # Main model
        #

        ts_input = layers.Input(self.ts_input_shape, name='ts_input')
        x = ts_input
        alt_out = tf.Variable(initial_value=1, trainable=False)

        # Select the correct parameters based on the selected ArchitectureVariant.
        if self.hyper.architecture_variant in [ArchitectureVariant.STGCN_1D, ArchitectureVariant.STGCN_1D_Grouped]:
            params = list(zip(self.hyper.cnn_1D_filters,
                              self.hyper.cnn_1D_filter_sizes,
                              self.hyper.cnn_1D_dilation_rates))
        elif self.hyper.architecture_variant == ArchitectureVariant.STGCN_2D:
            params = list(zip(self.hyper.cnn_2D_filters,
                              self.hyper.cnn_2D_filter_sizes,
                              self.hyper.cnn_2D_dilation_rates))

        else:
            raise NotImplementedError('Unknown ArchitectureVariant.')

        # Split the cnn parameters into parts for each block.
        params = np.split(np.array(params), self.hyper.nbr_st_blocks, axis=0)
        graph_model = None

        for block_id, params in enumerate(params):

            # Create st blocks based on the selected ArchitectureVariant.
            if self.hyper.architecture_variant in [ArchitectureVariant.STGCN_1D, ArchitectureVariant.SEQUENTIAL]:
                x, a_out, graph_model = st_block_1d(x, a, self.hyper, block_id, params, graph_model=graph_model)
            elif self.hyper.architecture_variant == ArchitectureVariant.STGCN_1D_Grouped:
                x, a_out, graph_model = st_block_1d_grouped(x, a, self.hyper, block_id, params, graph_model=graph_model)
            elif self.hyper.architecture_variant == ArchitectureVariant.STGCN_2D:
                x, a_out, graph_model = st_block_2d(x, a, self.hyper, block_id, params, graph_model=graph_model)

            # If weights between gcn layers should not be shared reset the reference to the graph layer
            # which is passed to the next ST block. This will result in a new graph layer instance being created.
            if not Modification.SHARE_GRAPH_WEIGHTS in self.hyper.modifications:
                graph_model = None

        pred_out, alt_out = output_block(x, self.hyper, self.dataset.num_classes, alt_out)

        return tf.keras.models.Model(inputs=[ts_input, a_input], outputs=[pred_out, a_out, alt_out], name='STGCN')

    def get_uncompiled_seq_model(self):

        #
        # Graph structure learning
        #

        a_input = layers.Input(self.a_input_shape, name='a_input', sparse=False)
        a = gsl_block(a_input, self.hyper, self.config, self.dataset)

        #
        # Main model
        #

        ts_input = layers.Input(self.ts_input_shape, name='ts_input')
        x, residual_source = ts_input, ts_input
        alt_out = tf.Variable(initial_value=1, trainable=False)

        params = list(zip(self.hyper.cnn_1D_filters,
                          self.hyper.cnn_1D_filter_sizes,
                          self.hyper.cnn_1D_dilation_rates))

        # Create a single st block as base for the sequential model.
        # Switching between standard 1d and grouped 1d must be done manually for the sequential model.

        # print('Sequential model uses standard 1D Convs.')
        # x, a_out, graph_model = st_block_1d(x, a, self.hyper, 0, params, graph_model=None)

        print('Sequential model uses grouped 1D Convs.')
        x, a_out, graph_model = st_block_1d_grouped(x, a, self.hyper, 0, params, graph_model=None)

        # If weights between gcn layers should not be shared reset the reference to the graph layer
        # which is passed to the next ST block
        if not Modification.SHARE_GRAPH_WEIGHTS in self.hyper.modifications:
            graph_model = None

        # Add additional graph layers after the base st block in order to match the no-sequential variant of the model.
        # Note that the iteration starts with 1 because the st-block already contains one graph layer.
        for id in range(1, self.hyper.nbr_st_blocks):
            x, a_out, graph_model = graph_layer(x, a, self.hyper, id, graph_model=graph_model)

            if not Modification.SHARE_GRAPH_WEIGHTS in self.hyper.modifications:
                graph_model = None

            # Add an activation function after the graph layer except for the last one.
            if id < self.hyper.nbr_st_blocks:
                x = get_activation_instance(self.hyper.activation, name=f'final_act_{id}')(x)

        x = tf.add(x, residual_source, name=f'res_con')
        x = get_activation_instance(self.hyper.activation, name=f'after_res_con_act')(x)

        pred_out, alt_out = output_block(x, self.hyper, self.dataset.num_classes, alt_out)

        return tf.keras.models.Model(inputs=[ts_input, a_input], outputs=[pred_out, a_out, alt_out], name='STGCN')

    def get_uncompiled_dev_model(self):

        #
        # Graph structure learning

        a_input = layers.Input(self.a_input_shape, name='a_input', sparse=False)
        a = gsl_block(a_input, self.hyper, self.config, self.dataset)

        #
        # Main model

        ts_input = layers.Input(self.ts_input_shape, name='ts_input')
        x = ts_input
        alt_out = tf.Variable(initial_value=1, trainable=False)
        pred_out = tf.Variable(initial_value=1, trainable=False)

        return tf.keras.models.Model(inputs=[ts_input, a_input], outputs=[pred_out, a, alt_out], name='DEV')

    def get_prediction_model(self, model: tf.keras.Model):

        """
        Appends a softmax layer to the main model so that the outputs are interpretable
        :param model:
        :return:
        """
        outputs = model.outputs
        pred_out = model.get_layer('pred_out').get_output_at(0)
        pred_out_softmax = tf.keras.layers.Softmax()(pred_out)
        outputs[0] = pred_out_softmax

        return tf.keras.models.Model(model.inputs, outputs)
