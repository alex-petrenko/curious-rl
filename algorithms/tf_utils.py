import tensorflow as tf


def dense(x, layer_size, regularizer=None, activation=tf.nn.relu):
    return tf.contrib.layers.fully_connected(
        x,
        layer_size,
        activation_fn=activation,
        weights_regularizer=regularizer,
        biases_regularizer=regularizer,
    )


def conv(x, num_filters, kernel_size, stride=1, regularizer=None, scope=None):
    return tf.contrib.layers.conv2d(
        x,
        num_filters,
        kernel_size,
        stride=stride,
        weights_regularizer=regularizer,
        biases_regularizer=regularizer,
        scope=scope,
    )


def count_total_parameters():
    """
    Returns total number of trainable parameters in the current tf graph.
    https://stackoverflow.com/a/38161314/1645784
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters
