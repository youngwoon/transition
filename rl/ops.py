import tensorflow as tf


def activation(activation):
    if activation == 'relu':
        return tf.nn.relu
    elif activation == 'elu':
        return tf.nn.elu
    elif activation == 'leaky':
        return tf.contrib.keras.layers.LeakyReLU(0.2)
    elif activation == 'tanh':
        return tf.tanh
    elif activation == 'sigmoid':
        return tf.sigmoid
    else:
        raise NotImplementedError('{} is not implemented'.format(activation))
