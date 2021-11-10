import numpy as np
import spektral
import tensorflow as tf


class MyInitializer(tf.keras.initializers.Initializer):

    def __init__(self, w):
        self.w = w

    def __call__(self, shape, dtype=None, **kwargs):
        print('Requested filter shape:', shape)

        # nodes x channel
        return self.w


A_in = np.array([
    [
        [0, 0, 0],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ],

], dtype='float32')

attn_kernel = np.array([[0], [1], [1], [0], [0]], dtype='float32')

# X = A_in
X = tf.keras.layers.Permute(dims=[2, 1], name='pre_agg_perm')(A_in)

att_sum = spektral.layers.GlobalAttnSumPool(attn_kernel_initializer=MyInitializer(attn_kernel))(X)

print(X.shape, attn_kernel.shape)
print(X)
print()
print(attn_kernel)
print('.........')

attn_coeff = np.dot(X, attn_kernel)
print('X*a_weights')
print(attn_coeff)

attn_coeff = np.squeeze(attn_coeff, -1)
print('X*a_weights squeezed')
print(attn_coeff)

attn_coeff = tf.keras.layers.Softmax()(attn_coeff)
print('softmax, axis = last')
print(attn_coeff)

att_sum_self = np.dot(attn_coeff, X)
print('#######################')
print(att_sum)
print(att_sum_self)
