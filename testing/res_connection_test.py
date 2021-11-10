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


def readable(array):
    return tf.transpose(np.squeeze(array), perm=[1, 0])


# equal to 1d
ts_example = [
    [
        [1, 2, 1, 1, 1, 1, 2, 1],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, -1, -1, -1, 1, -1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ]
]

ts_example = np.array(ts_example, dtype='float32')
ts_example = tf.transpose(ts_example, perm=[0, 2, 1])

model = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=(ts_example.shape[1], ts_example.shape[2])),
    tf.keras.layers.Reshape(target_shape=(-1, ts_example.shape[2], 1)),
    # Note the change of padding to same in order for res connection to work
    tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 1), strides=1, padding='same', dilation_rate=(2, 1),
                           use_bias=False, kernel_initializer=MyInitializer()),
    # Bei 1x1 macht dialation eh keinen sinn
    tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding='same', dilation_rate=1, use_bias=False,
                           kernel_initializer=tf.keras.initializers.Constant(value=1))
])

model.summary()
print()

out = model(ts_example)

# remove last dimension which is one, could be replaced by squease if for batches > 1
out = tf.reshape(out, shape=(1, -1, ts_example.shape[2]))
res = ts_example + out

print('x')
print(ts_example)
print()
print('f(x)')
print(out)
print()
print('h(x)')
print(res)
print()
print(out.shape, ts_example.shape, res.shape)
