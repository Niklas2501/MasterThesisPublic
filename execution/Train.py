import logging
import os
import sys

# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from stgcn.STGCN import STGCN
from configuration.Configuration import Configuration
from stgcn.Dataset import Dataset


def main():
    config = Configuration()

    if config.tf_memory_growth_enabled:
        import tensorflow as tf
        gpu = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset = Dataset(config)
    dataset.load()

    stgcn = STGCN(config, dataset, training=True)
    stgcn.print_model_info()
    stgcn.train_model()


if __name__ == '__main__':
    main()
