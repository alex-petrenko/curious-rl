import tensorflow as tf


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
