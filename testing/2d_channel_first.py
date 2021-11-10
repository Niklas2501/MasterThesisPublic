import numpy as np
import tensorflow as tf

tf.random.set_seed(1)


class MyInitializer(tf.keras.initializers.Initializer):

    def __init__(self):
        pass

    def __call__(self, shape, dtype=None, **kwargs):
        print('Requested filter shape:', shape)
        return np.array([
            [[[0.5, 1]]],
            [[[0.5, 1]]],
            [[[0.5, 1]]],
        ])


class MyChannelFirstInitializer(tf.keras.initializers.Initializer):

    def __init__(self):
        pass

    def __call__(self, shape, dtype=None, **kwargs):
        print('Requested filter shape:', shape)
        # return np.array([
        #     [[[0.5, 1]]],
        #     [[[0.5, 1]]],
        #     [[[0.5, 1]]],
        # ])
        return np.array([
            [
                [[0.5, 1]],
                [[0.5, 1]],
                [[0.5, 1]],
            ]
        ])


def readable(array):
    return tf.transpose(np.squeeze(array), perm=[1, 0])


ts_example = [
    [
        [1, 2, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, -1, -1, -1, 1, -1],
    ]
]

ts_example = np.array(ts_example)

#
# Working but not recommended
#

model = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=(ts_example.shape[1], ts_example.shape[2])),
    tf.keras.layers.Reshape(target_shape=(1, ts_example.shape[1], ts_example.shape[2])),

    # Note that the kernel size is switched
    tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 3), strides=1, padding='valid', dilation_rate=1,
                           use_bias=False, data_format='channels_first',
                           kernel_initializer=MyChannelFirstInitializer()),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='valid', dilation_rate=1,
                           use_bias=False, data_format='channels_first',
                           kernel_initializer=tf.keras.initializers.Constant(value=1)),
    # Just for output test
    # tf.keras.layers.Permute([3, 2, 1])
])

model.summary()
print()

out = model(ts_example)
out = np.squeeze(out)

print('Input shape, default time series config:')
print(np.squeeze(ts_example).shape)
print()
print(readable(ts_example))
print()
print('First step: Conv2D with n filters, kernel_size x,1')
print('First dim = time series length +- kernel size change through valid = no padding')
print('Second dim = time series depth --> unchanged because of kernel_size ,1')
print('Last dim = n, all others stay the same (with valid padding)')
print(out.shape)
print()
print(out)
print()
