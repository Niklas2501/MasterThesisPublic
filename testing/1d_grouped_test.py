import numpy as np
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


ts_example = [
    [
        [1, 2, 1, 1, 1, 1],
        [0, 0, 1, 0, 1, 0],
        [0, -1, -1, -1, 1, -1],
    ]
]

ts_example = np.array(ts_example, dtype='float32')
ts_example = tf.transpose(ts_example, perm=[0, 2, 1])

nbr_filters_per_group = 2
nbr_groups = 3

model = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=(ts_example.shape[1], ts_example.shape[2])),
    tf.keras.layers.Conv1D(filters=nbr_groups * nbr_filters_per_group, kernel_size=3, strides=1, padding='valid',
                           dilation_rate=1,
                           groups=nbr_groups,
                           use_bias=False,
                           # kernel_initializer=tf.keras.initializers.Constant(value=1))
                           kernel_initializer=MyInitializer())
])

model.summary()
print()

out = model(ts_example)

out_print = np.squeeze(out)

print('Input shape, default time series config:')
print(np.squeeze(ts_example).shape)
print()
print(readable(ts_example))
print()
print('First step: Grouped1D with n filters, kernel_size x,1')
print('First dim = time series length +- kernel size change through valid = no padding')
print('Second dim = time series depth --> unchanged because of kernel_size ,1')
print('Last dim = n, all others stay the same (with valid padding)')
print(out_print.shape)
print()
print(out_print)
print()

batch_size = 1

new_shape = (batch_size, -1, nbr_groups, nbr_filters_per_group)
# new_shape = (out.shape[0], out.shape[1], nbr_groups, -1)

out_new = tf.reshape(out, new_shape)
out_new_print = np.squeeze(out_new)
print(out_new_print)
print(out_new_print.shape)
print()

out_new_pooled = tf.math.reduce_sum(out_new, axis=3)
# out_new_pooled =  tf.keras.layers.Maximum()(out_new)
out_new_pooled_print = np.squeeze(out_new_pooled)
print(out_new_pooled_print)
print(out_new_pooled_print.shape)

print('----------------------------------------------------------------------------------------------')

# Same ops combined in single model:
# WARNING: Not adapted to change of axis
model = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=(ts_example.shape[1], ts_example.shape[2])),
    tf.keras.layers.Conv1D(filters=nbr_groups * nbr_filters_per_group,
                           groups=nbr_groups,
                           kernel_size=3, strides=1, padding='valid',
                           dilation_rate=1, use_bias=False, kernel_initializer=MyInitializer()),
    tf.keras.layers.Reshape(target_shape=(-1, nbr_groups, nbr_filters_per_group)),

    # https://stackoverflow.com/q/55510586/14648532
    tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=False))
])

out = model(ts_example)
out_print = out  # np.squeeze(out)

print(out_print)
print(out_print.shape)

print('----------------------------------------------------------------------------------------------')
# res connection test


model = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=(ts_example.shape[1], ts_example.shape[2])),
    tf.keras.layers.Conv1D(filters=nbr_groups * nbr_filters_per_group,
                           groups=nbr_groups,
                           kernel_size=3, strides=1, padding='same',
                           dilation_rate=2, use_bias=False, kernel_initializer=MyInitializer()),
    tf.keras.layers.Reshape(target_shape=(-1, nbr_groups, nbr_filters_per_group)),

    # https://stackoverflow.com/q/55510586/14648532
    # you can use K.mean instead of tf.reduce_mean and it would work. I used tf.reduce_mean because the OP used it in the question
    tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=False))
])

out = model(ts_example)
out_print = out  # np.squeeze(out)

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
