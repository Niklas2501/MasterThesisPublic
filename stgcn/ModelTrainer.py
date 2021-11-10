import os
import shutil
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from configuration.Configuration import Configuration
from configuration.Enums import GNN_Type, A_Variant, Modification
from configuration.Hyperparameter import Hyperparameter
from stgcn.Dataset import Dataset
from stgcn.InputGenerator import InputGenerator


class ModelTrainer:

    def __init__(self, config: Configuration, hyper: Hyperparameter, dataset: Dataset, model: tf.keras.Model):
        self.config = config
        self.hyper = hyper
        self.dataset = dataset
        self.model = model
        self.input_generator = InputGenerator(config, hyper, dataset)

        self.selected_model_name, self.selected_model_path = None, None
        self.training_model_path, self.checkpoint_path = None, None
        self.model_name_date_part = datetime.now().strftime("%m-%d_%H-%M-%S/")
        prefix = self.config.hyper_file.split('/')[-1].split('.')[0]
        self.set_file_names(prefix)

    def set_file_names(self, prefix):
        old_path = self.selected_model_path

        self.selected_model_name = prefix + '_' + self.model_name_date_part
        self.selected_model_path = self.config.models_folder + self.selected_model_name
        self.training_model_path = self.config.models_folder + 'temp_' + self.model_name_date_part
        self.checkpoint_path = self.training_model_path + 'weights-{epoch:d}'

        if old_path is not None and old_path != self.selected_model_path:
            os.rmdir(old_path)

        if not os.path.exists(self.selected_model_path):
            os.makedirs(self.selected_model_path)

        return self.selected_model_name

    def train_model(self, skip_model_saving=False):
        """
        Method for execution the model training process
        """

        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
        callbacks = [tf.keras.callbacks.TerminateOnNaN()]

        if not skip_model_saving:
            callbacks.append(tf.keras.callbacks.CSVLogger(self.selected_model_path + 'training.log'))

        if Modification.ADAPTIVE_LR in self.hyper.modifications:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1,
                                                                  factor=0.1, min_delta=0.0001, cooldown=1,
                                                                  min_lr=1e-8))

        if self.config.create_checkpoints and not skip_model_saving:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, monitor='val_loss',
                                                                verbose=1, save_best_only=True, save_weights_only=True))

        if self.hyper.early_stopping_enabled:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.hyper.early_stopping_limit,
                                                 restore_best_weights=True, verbose=1))

        train_input, y_train, val_data = self.input_generator.get_training_data()

        history = self.model.fit(
            x=train_input,
            y=y_train,
            epochs=self.hyper.epochs,
            batch_size=self.hyper.batch_size,
            callbacks=callbacks,
            verbose=2,
            shuffle=False,  # IMPORTANT
            validation_data=val_data,
            sample_weight=None
        )

        if not skip_model_saving:
            a_data, vis_out = self.get_aux_output(val_data)
            dense_weights = self.model.get_layer('pred_out').get_weights()

            self.save_model()
            self.create_plots(history.history, a_data, vis_out, dense_weights)

        return self.selected_model_name

    def get_aux_output(self, val_data):
        _, a_out, vis_out = self.model.predict(val_data[0])

        # pre_dense_out = None
        # pred_out , a_out = self.model.predict(val_data[0])

        # if self.hyper.gnn_type == GNN_Type.GAT and self.config.gat_plot_avg_all:
        #     a_out = np.squeeze(a_out)
        #     mean = np.mean(a_out, axis=0)
        #     a_data = {'ADJ': {'data': mean}}

        # A out has 4 dimensions if GAT layer is used. Check must be adapted for other A variants
        if len(a_out.shape) == 4:
            a_data = {}
            labels = self.dataset.get_train_val().get_y_strings()
            unique_labels = np.unique(labels)

            for i in range(self.hyper.gnn_layer_parameter):
                a_data[i] = {}

                for label in unique_labels:
                    indices_with_label = np.nonzero(labels == label)[0]
                    a_label = a_out[indices_with_label, :, i, :]
                    mean = np.mean(a_label, axis=0)
                    mean = np.squeeze(mean)
                    a_data[i][label] = {'data': mean}

                    if self.config.write_a_out_to_file:
                        sub_path = self.selected_model_path + '/a_out/'
                        if not os.path.exists(sub_path):
                            os.makedirs(sub_path)

                        np.save(sub_path + f'a_out_{label}_{i}.npy', mean)

        else:
            a_data = {'ADJ': {'data': a_out[0]}}

            if self.config.write_a_out_to_file:
                np.save(self.selected_model_path + 'a_out.npy', a_out[0])

            # Also store a plot of the non optimized input / initialisation values for comparison
            if self.hyper.a_variant in [A_Variant.A_OPT, A_Variant.A_PARAM]:
                a_data['ADJ_init'] = {'data': val_data[0][1][0]}

        return a_data, vis_out

    def save_model(self):

        # Deprecation warnings can be ignored: https://github.com/tensorflow/tensorflow/issues/44178

        # Variant 1 - Full model is stored, would be preferred but not possible due to usage of graph layer
        # self.model.save(self.selected_model_path + 'weights')

        # Variant 2 - Only weights are saved
        self.model.save_weights(self.selected_model_path + 'weights')

        # For this reason we need to store the hyperparameters, so we can recreate the same model before loading weights
        self.hyper.write_to_file(self.selected_model_path + "hyperparameters_used.json")

        # Delete the folder with the temporary models if enabled
        if self.config.create_checkpoints and self.config.delete_temp_models_after_training:
            shutil.rmtree(self.training_model_path, ignore_errors=False)
            print('\nDeleted temporary files in {}'.format(self.training_model_path))

        # Store the name of the selected model in the current config object
        # if it is reloaded from disk for some reason, should be used directly in most cases
        self.config.change_current_model(self.selected_model_name)

        print('\nLocation of saved model:', self.selected_model_path, '\n')

    def create_plots(self, history, a_data, vis_out, dense_weights):
        print('\nGenerating plots ...\n')

        if not os.path.exists(self.selected_model_path):
            os.makedirs(self.selected_model_path)

        # rankdir: 'TB' creates a vertical plot; 'LR' creates a horizontal plot.
        tf.keras.utils.plot_model(self.model, self.selected_model_path + "model.png",
                                  show_shapes=True, dpi=300, rankdir='TB')

        if self.config.write_vis_out_to_file:
            np.save(self.selected_model_path + 'vis_out.npy', vis_out)

        self.create_history_plot(history)

        if self.config.create_a_plots:
            self.create_a_plot(a_data)
        # self.cam_plot(dense_weights, pre_dense_out)

    def create_history_plot(self, history: dict):

        # Remove redundant data as long as no loss for A is integrated
        # history.pop('pred_out_loss')
        # history.pop('val_pred_out_loss')

        plt.set_cmap('jet')
        fig, (loss_subplot, metric_subplot, metric_detail) = plt.subplots(nrows=3, ncols=1, figsize=(30, 20),
                                                                          dpi=300, sharex='all')

        epochs = np.arange(0, len(history['loss']))

        for measure, values in history.items():
            if 'loss' in measure:
                loss_subplot.plot(epochs, values, label=measure)
            else:
                metric_subplot.plot(epochs, values, label=measure)
                metric_detail.plot(epochs, values, label=measure)

        loss_subplot.set_ylabel('Loss')
        loss_subplot.legend(loc='lower left')

        metric_subplot.set_xlabel('Epoch')
        metric_subplot.set_ylabel('Metric score')
        metric_subplot.legend(loc='lower left')

        metric_detail.set_xlabel('Epoch')
        metric_detail.set_ylabel('Metric score')
        metric_detail.legend(loc='lower left')
        metric_detail.set_ylim([0.95, 1.01])

        fig.savefig(self.selected_model_path + "history.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def create_a_plot(self, a_data):
        plt.set_cmap('hot_r')

        path = self.selected_model_path
        path = path + 'a_plots/'
        os.mkdir(path)

        if self.hyper.gnn_type == GNN_Type.GAT:
            for i in range(self.hyper.gnn_layer_parameter):

                if self.hyper.gnn_layer_parameter > 1:
                    path_head = path + f'head_{i}/'
                    os.mkdir(path_head)
                else:
                    path_head = path

                self.plot_group(a_data[i], path_head, i)

        else:
            self.plot_group(a_data, path)

    def plot_group(self, group, path, i=None):
        for key, values in group.items():
            a = values['data']

            size = 22 if self.config.a_plot_display_labels else 15
            dpi = 200 if self.config.a_plot_display_labels else 300

            vmin, vmax = (0, 1) if i is not None else (np.min(a), np.max(a))
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(size, size), dpi=dpi)
            im = ax.imshow(a, vmin=vmin, vmax=vmax)

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.5)
            plt.colorbar(im, cax=cax)

            ax.set_xlabel('j (source)')
            ax.set_ylabel('i (target)')

            ax.tick_params(which='minor', width=0)
            ax.set_xticks(np.arange(-.5, a.shape[0], 10), minor=True)
            ax.set_yticks(np.arange(-.5, a.shape[0], 10), minor=True)

            # Gridlines based on minor ticks
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.75)

            if self.config.a_plot_display_labels:
                # Minor ticks with width = 0 so they are not really visible
                ax.set_xticks(np.arange(0, a.shape[0], 1), minor=False)
                ax.set_yticks(np.arange(0, a.shape[0], 1), minor=False)

                features = [f[0:20] if len(f) > 20 else f for f in self.dataset.feature_names]

                ax.set_xticklabels(features, minor=False)
                ax.set_yticklabels(features, minor=False)

                # Rotate the tick labels and set their alignment.
                plt.setp(ax.get_xticklabels(), rotation=75, ha="right", rotation_mode="anchor")

            fig.tight_layout()

            if i is not None:
                fig.savefig(path + f"h_{i}_a_{key}_plot.pdf", dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(path + f"a_{key}_plot.pdf", dpi=dpi, bbox_inches='tight')
            plt.close(fig)
