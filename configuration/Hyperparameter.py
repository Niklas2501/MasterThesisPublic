import json

from configuration.Enums import *


class Hyperparameter:

    def __init__(self):
        ##
        # Important: Variable names must match json file entries
        ##

        ##
        # General settings
        ##

        # Need to be changed after dataset was loaded
        self.time_series_length = None
        self.time_series_depth = None

        self.notes = None

        self.spacer0 = '###################### General hyperparameters for all models ######################'

        # Selection which basic architecture is used, see enum class for details
        self.architecture_variant = None
        self.modifications = None

        self.batch_size = None
        self.epochs = None
        self.learning_rate = None
        self.dropout_rate = None

        self.early_stopping_enabled = None
        self.early_stopping_limit = None

        self.gradient_cap_enabled = None
        self.gradient_cap = None

        self.loss_function = None

        self.spacer1 = '###################### Layer configuration ######################'

        self.cnn_1D_filters = None
        self.cnn_1D_filter_sizes = None
        self.cnn_1D_padding = None
        self.cnn_1D_dilation_rates = None

        self.cnn_2D_filters = None
        self.cnn_2D_filter_sizes = None
        self.cnn_2D_dilation_rates = None

        self.spacer2 = '############# General model configuration #############'
        self.activation = None
        self.a_variant = None
        self.permute_before_agg = None
        self.final_agg = None

        self.spacer3 = '############# ST block configuration #############'
        self.gnn_type = None
        self.gnn_layer_parameter = None
        self.var_parameter = None
        self.nbr_st_blocks = None
        self.conv_after_graph = None
        self.res_connection = None
        self.a_emb_alpha = None
        self.a_emb_dense_dim = None
        self.loop_weight = None

        ##
        # Class variables that are not stored (name must start with __)
        ##
        self.__is_loaded = False

    def is_loaded(self):
        return self.__is_loaded

    def set_time_series_properties(self, time_series_length, time_series_depth):
        self.time_series_length = time_series_length
        self.time_series_depth = time_series_depth

    # Allows the import of a hyper parameter configuration from a json file
    def load_from_file(self, file_path, use_hyper_file=True):
        self.__is_loaded = True

        if not use_hyper_file:
            return

        file_path = file_path + '.json' if not file_path.endswith('.json') else file_path

        with open(file_path, 'r') as f:
            data = json.load(f)

        ##
        # General settings
        ##

        self.notes = data['notes']

        ##
        # General hyperparameters for all models
        ##

        self.architecture_variant = ArchitectureVariant(data['architecture_variant']) if data[
                                                                                             'architecture_variant'] is not None else None
        self.modifications = [Modification(mod) for mod in data['modifications']] if data[
                                                                                         'modifications'] is not None else None

        self.batch_size = data['batch_size']
        self.epochs = data['epochs']
        self.learning_rate = data['learning_rate']
        self.dropout_rate = data['dropout_rate']
        self.loss_function = LossFunction(data['loss_function']) if data['loss_function'] is not None else None

        self.early_stopping_enabled = data['early_stopping_enabled']
        self.early_stopping_limit = data['early_stopping_limit']

        self.gradient_cap_enabled = data['gradient_cap_enabled']
        self.gradient_cap = data['gradient_cap']

        ##
        # Layer configuration
        ##

        self.cnn_1D_filters = data['cnn_1D_filters']
        self.cnn_1D_filter_sizes = data['cnn_1D_filter_sizes']
        self.cnn_1D_padding = Padding(data['cnn_1D_padding']) if data['cnn_1D_padding'] is not None else None
        self.cnn_1D_dilation_rates = data['cnn_1D_dilation_rates']

        self.cnn_2D_filters = data['cnn_2D_filters']
        self.cnn_2D_filter_sizes = data['cnn_2D_filter_sizes']
        self.cnn_2D_dilation_rates = data['cnn_2D_dilation_rates']

        ##
        # General model configuration
        ##

        self.activation = data['activation']
        self.a_variant = A_Variant(data['a_variant']) if data['a_variant'] is not None else None
        self.permute_before_agg = data['permute_before_agg']
        self.final_agg = FinalAggregation(data['final_agg']) if data['final_agg'] is not None else None

        ##
        # ST block configuration
        ##

        self.gnn_type = GNN_Type(data['gnn_type']) if data['gnn_type'] is not None else None
        self.gnn_layer_parameter = data['gnn_layer_parameter']
        self.var_parameter = data['var_parameter']
        self.nbr_st_blocks = data['nbr_st_blocks']
        self.conv_after_graph = data['conv_after_graph']
        self.res_connection = ResConnection(data['res_connection']) if data['res_connection'] is not None else None
        self.a_emb_alpha = data['a_emb_alpha']
        self.a_emb_dense_dim = data['a_emb_dense_dim']
        self.loop_weight = data['loop_weight']

    def write_to_file(self, path_to_file):
        with open(path_to_file, 'w') as outfile:
            json.dump(self.get_dict(), outfile, sort_keys=False, indent=4)

    def print_hyperparameters(self):
        print(json.dumps(self.get_dict(), sort_keys=False, indent=4))

    def get_dict(self):
        return {key: value for key, value in self.__dict__.items() if
                not (key.startswith('__') or key.startswith('_Hyperparameter__')) and not callable(key)}

    def check_constrains(self):
        if not self.is_loaded():
            raise ValueError('Not loaded.')

        required_hps = ['architecture_variant', 'batch_size', 'epochs', 'learning_rate', 'early_stopping_enabled',
                        'early_stopping_limit', 'gradient_cap_enabled', 'gradient_cap', 'loss_function',
                        'modifications', 'activation', 'a_variant', 'permute_before_agg', 'final_agg', 'gnn_type',
                        'nbr_st_blocks', 'res_connection']

        for hp in required_hps:
            hp_value = getattr(self, hp)
            if hp_value is None:
                raise ValueError(f'Required parameter {hp} is not set.')

        # Relation between number of blocks and number of cnn layers configured
        parts = []
        if self.architecture_variant == ArchitectureVariant.STGCN_1D:
            parts = [self.cnn_1D_filters, self.cnn_1D_filter_sizes, self.cnn_1D_dilation_rates]
        elif self.architecture_variant == ArchitectureVariant.STGCN_2D:
            parts = [self.cnn_2D_filters, self.cnn_2D_filter_sizes, self.cnn_2D_dilation_rates]

        # Skip if check is not necessary for architecture variant
        if ArchitectureVariant.block_based_architecture_selected(self.architecture_variant) and len(parts) > 0:
            nbr_cnn_required = self.nbr_st_blocks * (2 if self.conv_after_graph else 1)
            for part in parts:

                # Condition for multiple temporal layers before and after the graph layer in each block.
                if not len(part) % nbr_cnn_required == 0 or len(parts[0]) != len(part):
                    raise ValueError('Convolutional parameters do not match the block configuration.')

                # Condition for single temporal layers
                # if not len(part) == nbr_cnn_required:
                #     raise ValueError(
                #         f'The length of some convolution parameter does not match the STGCN block configuration.'
                #         f'\nFor {self.nbr_st_blocks} blocks and conv_after_graph={self.conv_after_graph} '
                #         f'{nbr_cnn_required} convolution layers are needed.')

        if self.architecture_variant == ArchitectureVariant.SEQUENTIAL:
            if not self.res_connection == ResConnection.SEQ:
                raise ValueError('ResConnection.SEQ must be used for a sequential model.')

            if self.conv_after_graph:
                raise ValueError('Temporal conv. after the graph layer must be disabled for a sequential model.')

        if Modification.L1_REG in self.modifications and Modification.KNN_RED in self.modifications:
            raise ValueError('L1_REG and KNN_RED can not be used at the same time.')

        if Modification.KNN_RED in self.modifications and self.final_agg == FinalAggregation.GLOBAL_ATT:
            raise ValueError('Modification.KNN_RED and FinalAggregation.GLOBAL_ATT both use var_parameter.')

        if (self.architecture_variant == ArchitectureVariant.STGCN_1D_Grouped and not (
                Modification.GROUPED_G_1x1_AGG in self.modifications
                or Modification.GROUPED_1D_1x1_AGG in self.modifications)):
            raise ValueError('No 1x1 aggregation method set')

        warnings = []

        if self.final_agg == FinalAggregation.CUSTOM_FLATTEN and not self.permute_before_agg:
            warnings.append('Using custom flatten as final aggregation without permute_before_agg may result'
                            'in an unintended behaviour.')

        if self.a_variant == A_Variant.A_PRE and self.gnn_type != GNN_Type.GAT or (
                self.a_variant != A_Variant.A_PRE and self.gnn_type == GNN_Type.GAT):
            warnings.append('A_Pre variant should primarily be used with GNN_Type.GAT and vise versa.')

        if (Modification.L1_REG in self.modifications or Modification.KNN_RED in self.modifications) \
                and self.a_variant in [A_Variant.A_PRE, A_Variant.A_I, A_Variant.A_DEV]:
            warnings.append('L1_REG/KNN_RED is/should not used by the currently selected A variant.')

        if self.conv_after_graph and self.res_connection != ResConnection.V0:
            warnings.append('ResConnections != V0 are not intended to be used with conv_after_graph!')

        if Modification.GSL_RANDOM_INIT in self.modifications and not A_Variant.supports_random_init(self.a_variant):
            warnings.append('The selected A variant does not support the modification GSL_RANDOM_INIT.')

        if len(warnings) > 0:
            spacer = '##################################################' * 2
            print(spacer)
            print('CONFIGURATION WARNINGS:')
            print(spacer)
            for warning in warnings:
                print('\t- ' + warning)
            print(spacer, '\n')
