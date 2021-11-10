import logging
import os

# suppress debugging messages of TensorFlow
from stgcn.InputGenerator import InputGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import spektral
import tensorflow as tf
from tensorflow.keras import layers


def get_diff_conv_model(shape, sym):
    a_in = layers.Input(shape=shape[1:])

    # Do TF A Preprocessing here
    a = a_in

    if sym:
        pass

    else:
        pass

    ############################
    a_out = a

    model = tf.keras.Model(a_in, a_out)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy()
    )

    return model


def get_gcn_model(shape, sym, layer):
    a_in = tf.keras.Input(shape=shape[1:])

    # Do TF A Preprocessing here
    a = a_in

    if layer == 'gcn':
        a = InputGenerator.gcn_preprocessing(a, sym)
    elif layer == 'diff':
        a = InputGenerator.diff_preprocessing(a, sym)

    a_out = a
    ############################

    return tf.keras.Model(a_in, a_out)





def main():
    seed = 123
    test_sym = False
    test_random = False
    test_layer = ['gcn', 'diff'][0]

    a = np.array([
        # A, B, C, D
        [1, 1, 1, 0],  # A
        [1, 0, 0, 1],  # B
        [1, 0, 1, 0],  # C
        [0, 0, 1, 0]  # D
    ])

    a = np.expand_dims(a, 0)

    a_rand = np.random.default_rng(seed).random(4 * 4).reshape((1, 4, 4))
    a = a_rand if test_random else a.astype('float32')

    # a = np.array([
    #     np.random.default_rng(123).random(4 * 4).reshape((4, 4)),
    #     np.random.default_rng(456).random(4 * 4).reshape((4, 4))
    # ])

    if test_layer == 'gcn':
        a_s = spektral.utils.gcn_filter(a, test_sym)
    elif test_layer == 'diff':
        # a_s = DiffusionConv.preprocess(a, test_sym)
        # a_s = np.squeeze(np.array(a_s))
        pass
    else:
        raise ValueError()

    model = get_gcn_model(a.shape, test_sym, test_layer)
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy())
    model.summary()

    a_tf = model(a)

    if test_layer == 'gcn':

        ##########################
        a_tf_final = np.squeeze(a_tf[-1].numpy()).astype('float32')
        a_s = np.squeeze(a_s).astype('float32')

        a_s = a_s.round(5)
        a_tf_final = a_tf_final.round(5)

        print('---------------------')
        print('A')
        print(np.squeeze(a_tf[0].numpy()))
        print('A_I')
        print(np.squeeze(a_tf[1].numpy()))
        print('D')
        print(np.squeeze(a_tf[2].numpy()))
        print('D_pow')
        print(np.squeeze(a_tf[3].numpy()))
        print('A_hat')
        print(np.squeeze(a_tf[4].numpy()))
        print('----------------------')
        print(a_s)
        print()
        print(np.array_equal(a_s, a_tf_final))
    else:
        print(a_tf[0].numpy())
        print(a_tf[1].numpy())
        print(a_tf[0].shape)
        print(a_tf[1].shape)
        # a_s_f = a_s[0].astype('float32').round(5)
        # a_tf_f = np.squeeze(a_tf[0].numpy()).astype('float32').round(5)
        # a_s_b = a_s[1].astype('float32').round(5)
        # a_tf_b = np.squeeze(a_tf[1].numpy()).astype('float32').round(5)
        #
        # print('f')
        # print(a_s_f)
        # print()
        # print(a_tf_f)
        # print('b')
        # print(a_s_b)
        # print()
        # print(a_tf_b)
        # print()
        # print('f,equal', np.array_equal(a_s_f, a_tf_f))
        # print('b,equal', np.array_equal(a_s_b, a_tf_b))

    # degrees = np.power(np.array((a+ np.eye(a.shape[0])).sum(1)), -1).ravel()
    # print(degrees)


if __name__ == '__main__':
    main()
