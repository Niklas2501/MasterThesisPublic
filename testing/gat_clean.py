import numpy as np
import spektral
import tensorflow as tf

A = np.array([

    # No connection from or to A
    [
        # A, B, C, D
        [1, 0, 0, 0],  # A
        [0, 1, 0, 0],  # B
        [0, 0, 1, 0],  # C
        [0, 0, 0, 1]  # D
    ],

    # Connection A -> C
    [
        # A, B, C, D
        [1, 0, 1, 0],  # A
        [0, 1, 0, 0],  # B
        [0, 0, 1, 0],  # C
        [0, 0, 0, 1]  # D
    ],

    # Connection C -> A
    [
        # A, B, C, D
        [1, 0, 0, 0],  # A
        [0, 1, 0, 0],  # B
        [1, 0, 1, 0],  # C
        [0, 0, 0, 1]  # D
    ],

    # Connection A -> C and C -> A
    [
        # A, B, C, D
        [1, 0, 1, 0],  # A
        [0, 1, 0, 0],  # B
        [1, 0, 1, 0],  # C
        [0, 0, 0, 1]  # D
    ],
])

X = np.array([
    [
        [2, 1, 0],
        [0.5, -0.5, 1],
        [1, 2, 1.5],
        [1, 1, 0]
    ],
    [
        [2, 1, 0],
        [0.5, -0.5, 1],
        [1, 2, 1.5],
        [1, 1, 0]
    ],
    [
        [2, 1, 0],
        [0.5, -0.5, 1],
        [1, 2, 1.5],
        [1, 1, 0]
    ],
    [
        [2, 1, 0],
        [0.5, -0.5, 1],
        [1, 2, 1.5],
        [1, 1, 0]
    ],
])

W = np.array([
    [[1, 1]],
    [[1, 1]],
    [[1, 1]],
])

tf.random.set_seed(1)
n_nodes = X.shape[1]
n_features = X.shape[2]
n_channels = 2

x_input = tf.keras.Input(shape=(n_nodes, n_features))
a_input = tf.keras.Input(shape=(n_nodes, n_nodes), sparse=True)
layer = spektral.layers.GATConv(channels=n_channels, activation=None, use_bias=False,
                                kernel_initializer=tf.keras.initializers.Constant(value=1), attn_heads=1,
                                concat_heads=True, )

output = layer([x_input, a_input])
graph_model = tf.keras.Model([x_input, a_input], output, name=f'gat')

graph_model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy()
)

out = graph_model([X, A]).numpy()

print('No connection from or to A')
print(out[0])
print('Connection A -> C')
print(out[1])
print('Connection C -> A')
print(out[2])
print('Connection A -> C and C -> A')
print(out[3])
