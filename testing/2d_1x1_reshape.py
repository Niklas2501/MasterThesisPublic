import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class MyInitializer(tf.keras.initializers.Initializer):

    def __init__(self):
        pass

    def __call__(self, shape, dtype=None, **kwargs):
        print('Requested filter shape:', shape)
        return np.array([
            [[1, 0, 0, 0, 1, 0]],
            [[0, 1, 0, 0, 0, 1]],
            [[0, 0, 1, 0, 0, 0]]
        ])


np.random.seed(25011997)

nodes = 3
node_features = 6
filters_per_node = 2

input = np.random.random_integers(0, 3, nodes * node_features)
input = np.reshape(input, newshape=(1, node_features, nodes))
input = input.astype('float32')

print(input)

x = layers.Conv1D(filters=filters_per_node * nodes, groups=nodes, kernel_size=(3,), dilation_rate=(1,),
                  kernel_initializer=MyInitializer(),
                  use_bias=False,
                  padding='valid', name=f'temporal_cnn')(input)

print('grouped output')
print(x)
print(x.shape)
print()

# x = layers.Reshape(target_shape=(x.shape[1], nodes, filters_per_node), name=f'2d_reshape')(x)

print('reshaped output')
print(x)
print(x.shape)
print()

# x = layers.Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=(1, 1), padding='same', name=f'2d_1x1',
#                   kernel_initializer=tf.keras.initializers.Ones(),
#                   use_bias=False)(x)
x = layers.Conv1D(filters=1*nodes, groups=nodes, kernel_size=(1,), dilation_rate=(1,), padding='causal',
                  kernel_initializer=tf.keras.initializers.Ones(),
                  use_bias=False)(x)


print('2d output')
print(x)
print(x.shape)
print()

x = tf.squeeze(x, axis=-1, name=f'2d_squeeze')

print('grouped output')
print(x)
print(x.shape)
print()
