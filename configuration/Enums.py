from enum import Enum


class ArchitectureVariant(str, Enum):
    STGCN_1D = 'stgcn_1d'
    STGCN_1D_Grouped = 'stgcn_1d_grouped'
    STGCN_2D = 'stgcn_2d'
    SEQUENTIAL = 'sequential'
    DEV = 'dev'

    @staticmethod
    def block_based_architecture_selected(architecture):
        return architecture in [ArchitectureVariant.STGCN_1D,
                                ArchitectureVariant.STGCN_1D_Grouped,
                                ArchitectureVariant.STGCN_2D]


class Modification(str, Enum):
    # Forces the learned adjacency matrix to contain self loop (i.e. A_ii = 1).
    FORCE_IDENTITY = 'force_identity'

    # All graph layers in the stgcn model not only share the adjacency matrix but also their weights.
    SHARE_GRAPH_WEIGHTS = 'share_graph_weights'

    # Applies layer specific normalisation to the learned adjacency matrix,
    # i.e. converts A^l to the normalised laplacian  matrix.
    FORCE_NORMALISATION = 'force_normalisation'

    # Uses standard 1d 1x1 convolutions to aggregate the datastream wise features maps globally.
    GROUPED_1D_1x1_AGG = 'grouped_1d_1x1_agg'

    # Uses 2d 1x1 convolutions to aggregate the datastream wise features maps datastream wise.
    GROUPED_G_1x1_AGG = 'grouped_g_1x1_agg'

    # l1 regularisation of the learned adjacency matrix.
    L1_REG = 'l1_reg'

    # Reduced the learned adjacency matrix to the k strongest connections per node.
    KNN_RED = 'knn_red'

    # Learning rate will be reduced if a plateau is reached.
    ADAPTIVE_LR = 'adaptive_lr'

    # Randomly initialise the a learning approaches. Used for the ablation study .
    GSL_RANDOM_INIT = 'gsl_random_init'


class ResConnection(str, Enum):
    # V0 in thesis
    V0 = 'v0'

    # V2 in thesis
    V3 = 'v3'

    # V1 in thesis
    V4 = 'v4'

    # Variants not discussed in thesis due to bad performance.
    V1 = 'v1'
    V2 = 'v2'

    # In case a sequential model is created, no residual connection should be created because additional
    # graph layers might follow.
    SEQ = 'seq'


class LossFunction(str, Enum):
    CATEGORICAL_CROSSENTROPY = 'categorical_crossentropy'
    WCCE_V1 = 'wcce_v1'
    WCCE_V2 = 'wcce_v2'
    WCCE_V3 = 'wcce_v3'
    WCCE_V4 = 'wcce_v4'

    @staticmethod
    def is_wcce_variant(loss):
        return loss in [LossFunction.WCCE_V1, LossFunction.WCCE_V2, LossFunction.WCCE_V3, LossFunction.WCCE_V4]


class Padding(str, Enum):
    VALID = 'valid'
    CAUSAL = 'causal'
    SAME = 'same'


class BaselineAlgorithm(str, Enum):
    EUCLIDEAN_DISTANCE = 'euclidean_distance'
    RIDGE_CLASSIFIER = 'ridge_classifier'
    DTW = 'DTW'
    SNN = 'snn'

    @staticmethod
    def is_instance_based(algorithm):
        return algorithm not in [BaselineAlgorithm.RIDGE_CLASSIFIER, BaselineAlgorithm.SNN]


class SplitMode(str, Enum):
    # Examples of a single run to failure are not in both train and test
    ENSURE_NO_MIX = 'ensure_no_mix'

    # The standard train_test_split method of skit-learn is used.
    SEEDED_SK_LEARN = 'seeded_sk_learn'


class GridSearchMode(str, Enum):
    STANDARD_GRID_SEARCH = 'standard_grid_search'
    TEST_MULTIPLE_JSONS = 'test_multiple_jsons'
    INDEX_PAIR_GRID_SEARCH = 'index_pair_grid_search'


class DatasetPart(str, Enum):
    TRAIN = 'train'
    TEST = 'test'
    TRAIN_VAL = 'train_val'
    TEST_VAL = 'test_val'


class Dataset(str, Enum):
    FT_MAIN = 'FT_MAIN'
    FT_LEGACY = 'FT_LEGACY'

    @staticmethod
    def is_3rd_party(dataset):
        return dataset not in [Dataset.FT_MAIN, Dataset.FT_LEGACY]

    @staticmethod
    def is_wip_dataset(dataset):
        return dataset in [Dataset.FT_MAIN]

    @staticmethod
    def is_feature_vector_dataset(dataset):
        return dataset in []

    @staticmethod
    def uses_aux_df(dataset):
        return dataset in [Dataset.FT_MAIN, Dataset.FT_LEGACY]


class Representation(str, Enum):
    RAW = 'RAW'
    ROCKET = 'ROCKET'
    MINI_ROCKET = 'MINI_ROCKET'
    MULTI_ROCKET = 'MULTI_ROCKET'

    @staticmethod
    def contains_feature_vectors(rep):
        return rep in [Representation.ROCKET, Representation.MINI_ROCKET, Representation.MULTI_ROCKET]

    @staticmethod
    def get_all(exclude):
        return [rep for rep in
                [Representation.RAW, Representation.ROCKET,
                 Representation.MINI_ROCKET, Representation.MULTI_ROCKET] if rep not in exclude]


class A_Variant(str, Enum):
    A_PRE = 'a_pre'
    A_OPT = 'a_opt'
    A_PARAM = 'a_param'
    A_EMB_V1 = 'a_emb_v1'
    A_EMB_V2 = 'a_emb_v2'
    A_EMB_V3 = 'a_emb_v3'
    A_I = 'a_i'
    A_DEV = 'a_dev'

    @staticmethod
    def uses_predefined(variant):
        return variant in [A_Variant.A_PRE, A_Variant.A_OPT, A_Variant.A_PARAM]

    @staticmethod
    def is_emb_variant(variant):
        return variant in [A_Variant.A_EMB_V1, A_Variant.A_EMB_V2, A_Variant.A_EMB_V3]

    @staticmethod
    def placeholder_input(variant):
        return variant not in [A_Variant.A_PRE, A_Variant.A_OPT, A_Variant.A_I]

    @staticmethod
    def supports_random_init(variant):
        return variant in [A_Variant.A_EMB_V1, A_Variant.A_EMB_V2, A_Variant.A_EMB_V3, A_Variant.A_PARAM]


class GNN_Type(str, Enum):
    ARMA = 'arma'
    DIFF = 'diff'
    GAT = 'gat'
    GCN = 'gcn'


class FinalAggregation(str, Enum):
    GLOBAL_AVG = 'global_avg'
    GLOBAL_MAX = 'global_max'
    GLOBAL_SUM = 'global_sum'
    GLOBAL_ATT = 'global_att'
    GLOBAL_ATT_SUM = 'global_att_sum'
    CUSTOM_FLATTEN = 'custom_flatten'
    RNN_AGG = 'rnn'


class Activation(str, Enum):
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    ELU = 'elu'
