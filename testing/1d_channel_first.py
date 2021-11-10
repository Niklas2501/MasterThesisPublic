import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.random.set_seed(1)


class MyInitializer(tf.keras.initializers.Initializer):

    def __init__(self):
        pass

    def __call__(self, shape, dtype=None, **kwargs):
        print('Requested filter shape:', shape)
        return np.array([
            [[0, 1, 2, 1, 2, 0]],
            [[0, 1, 2, 1, 2, 0]],
            [[0, 1, 2, 1, 2, 0]]
        ])


def readable(array):
    return tf.transpose(np.squeeze(array), perm=[1, 0])


# equal to 1d
ts_example = [
    [
        [1, 2, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, -1, -1, -1, 1, -1],
    ]
]

ts_example = np.array(ts_example)
# ts_example = tf.transpose(ts_example, perm=[0, 2, 1])
# ts_example = tf.expand_dims(ts_example, axis=-2)

batch_size = 1
nbr_groups = 3
nbr_filters_per_group = 2

model = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=(ts_example.shape[1], ts_example.shape[2])),
    tf.keras.layers.Conv1D(filters=nbr_groups * nbr_filters_per_group,
                           groups=nbr_groups,
                           kernel_size=3, strides=1, padding='valid',
                           dilation_rate=1, use_bias=False, kernel_initializer=MyInitializer(),
                           data_format='channels_first'),
    # tf.keras.layers.Reshape(target_shape=(batch_size, nbr_groups, nbr_filters_per_group, -1)),

    # https://stackoverflow.com/q/55510586/14648532
    # tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=False))
])

print()
print('Input shape:', ts_example.shape)

out = model(ts_example)
out_print = np.squeeze(out)

print('CNN 1D Only')
print(out_print)
print(out_print.shape)
print()
print('------------------------------------------------------------')
print()

model = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=(ts_example.shape[1], ts_example.shape[2])),
    tf.keras.layers.Conv1D(filters=nbr_groups * nbr_filters_per_group,
                           groups=nbr_groups,
                           kernel_size=3, strides=1, padding='valid',
                           dilation_rate=1, use_bias=False, kernel_initializer=MyInitializer(),
                           data_format='channels_first'),
    tf.keras.layers.Reshape(target_shape=(nbr_groups, nbr_filters_per_group, -1)),
])

out = model(ts_example)
out_print = np.squeeze(out)

print('CNN1D + Reshape')
print(out_print)
print(out_print.shape)
print()
print('------------------------------------------------------------')
print()

model = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=(ts_example.shape[1], ts_example.shape[2])),
    tf.keras.layers.Conv1D(filters=nbr_groups * nbr_filters_per_group,
                           groups=nbr_groups,
                           kernel_size=3, strides=1, padding='valid',
                           dilation_rate=1, use_bias=False, kernel_initializer=MyInitializer(),
                           data_format='channels_first'),
    tf.keras.layers.Reshape(target_shape=(nbr_groups, nbr_filters_per_group, -1)),

    # https://stackoverflow.com/q/55510586/14648532
    # Shape before: (groups = original feature dim, filters per group, time steps)
    tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-2, keepdims=False)),
])

out = model(ts_example)
out_print = out  # np.squeeze(out)

print('CNN1D + Reshape + Reduce')
print(out_print)
print(out_print.shape)
print('In other format:')
print(np.swapaxes(np.squeeze(out_print), 0, 1))
print()
print('------------------------------------------------------------')
print()
