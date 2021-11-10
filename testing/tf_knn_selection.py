import logging
import os

import numpy as np

# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

k = 3
# @formatter:off
A_in = np.array([
    [
        [0,     7,      0,      0,      -2],
        [5,     6,      3,      -4,      -5],
        [3,     0,      1,      4,      -6],
        [-5,    -5,     -5,     0,      1],
        [0,     0,      0,      -6,     0],
    ]
], dtype='float32')
# @formatter:on

# A_in = np.expand_dims(A_in, axis=0)
print(A_in.shape)

from stgcn.ModelComponents import knn_reduction

output = knn_reduction(A_in, 2, convert_to_binary=False, based_on_abs_values=False)
output_abs = knn_reduction(A_in, 2, convert_to_binary=False, based_on_abs_values=True)
output_abs_bin = knn_reduction(A_in, 2, convert_to_binary=True, based_on_abs_values=True)

#
# model_in = tf.keras.Input(shape=(5, 5))
# A = model_in
#
# k_inverse = A.shape[1] - k
#
# A = tf.transpose(A, perm=[0, 2, 1])
# top_k = tf.math.top_k(A, k).values
# kth_largest = tf.reduce_min(top_k, axis=-1, keepdims=False)
# kth_largest = tf.transpose(kth_largest)
# A = tf.where(A < kth_largest, 0.0, 1.0)
# A = tf.transpose(A, perm=[0, 2, 1])

# A_temp = tf.math.negative(A_temp)
# A_temp = tf.math.top_k(A_temp, k=k_inverse).indices
# A= tf.scatter_nd(
#    indices, updates, shape, name=None
# )


# model = tf.keras.Model(model_in, A)

# output = model(A_in).numpy()

print(A_in)
print(output)
print(output_abs)
print(output_abs_bin)
