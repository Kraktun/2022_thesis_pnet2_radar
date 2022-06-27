import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential

def _regularizer_from_string(regularizer):
    """
    Simple way to create a regularizer from a dictionary with name and value
    """
    regularizer_name = regularizer['name']
    value = regularizer['value']
    assert regularizer_name in ['l1', 'l2', 'l1_l2'], 'Invalid regularizer'
    if regularizer_name == 'l1':
        my_reg = tf.keras.regularizers.L1(value)
    elif regularizer_name == 'l2':
        my_reg = tf.keras.regularizers.L2(value)
    elif regularizer_name == 'l1_l2':
        my_reg = tf.keras.regularizers.L1L2(*value)
    return my_reg

def parse_regularizers(source_args):
    # parse the regularizers described as strings + values and return the parameters
    args = source_args.copy()
    for k in args.keys():
        if "regularizer" in k:
            args[k] = _regularizer_from_string(args[k])
    return args

"""
These are all layers for Pointnet (the first version). They haven't been used in quite a lot of time, so don't expect
them to work (but maybe they do...).
"""

def getConv3DPointLayer(filters,
             kernel_size,
             stride=[1, 1, 1],
             padding='SAME',
             data_format='channels_last',
             use_xavier=True,
             stddev=1e-3,
             weight_decay=None,
             activation_fn="relu",
             batch_norm=False,
             batch_norm_decay=None,
             name=None):
    
    if use_xavier:
        kernel_initializer = tf.keras.initializers.GlorotNormal() # glorot normal == xavier
    else:
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)

    kernel_regularizer = None
    if weight_decay is not None:
        kernel_regularizer = tf.keras.regularizers.L2(weight_decay)

    all_layers = []
    cv = layers.Conv3D(filters,
                       kernel_size,
                       strides=stride,
                       padding=padding,
                       data_format=data_format, 
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_initializer=tf.constant_initializer(0.),
                       activation=activation_fn)
    all_layers.append(cv)
    
    if batch_norm:
        if data_format == "channels_last":
            axis = -1
        else:
            axis = 1
        bn = layers.BatchNormalization(axis=axis, momentum=batch_norm_decay)
        all_layers.append(bn)

    return Sequential(all_layers, name=name)


def getConv2DPointLayer(filters,
            kernel_size,
            stride=[1, 1],
            padding='SAME',
            data_format='channels_last',
            use_xavier=True,
            stddev=1e-3,
            weight_decay=None,
            activation_fn="relu",
            batch_norm=False,
            batch_norm_decay=None,
            name=None):
    
    if use_xavier:
        kernel_initializer = tf.keras.initializers.GlorotNormal() # glorot normal == xavier
    else:
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)

    kernel_regularizer = None
    if weight_decay is not None:
        kernel_regularizer = tf.keras.regularizers.L2(weight_decay)

    all_layers = []
    cv = layers.Conv2D(filters,
                        kernel_size,
                        stride,
                        padding=padding,
                        data_format=data_format, 
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_initializer=tf.constant_initializer(0.),
                        activation=activation_fn)
    all_layers.append(cv)
    
    if batch_norm:
        if data_format == "channels_last":
            axis = -1
        else:
            axis = 1
        bn = layers.BatchNormalization(axis=axis, momentum=batch_norm_decay)
        all_layers.append(bn)

    return Sequential(all_layers, name=name)

def get_conv3D_point_transpose_layer(filters,
        kernel_size,
        strides=[1, 1, 1],
        padding='VALID',
        data_format='channels_last',
        use_xavier=True,
        stddev=1e-3,
        weight_decay=None,
        activation_fn="relu",
        batch_norm=False,
        batch_norm_decay=None,
        name=None):
    # Note that batch normalization does not work currently
    if use_xavier:
        kernel_initializer = tf.keras.initializers.GlorotNormal() # glorot normal == xavier
    else:
        kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)

    kernel_regularizer = None
    if weight_decay is not None:
        kernel_regularizer = tf.keras.regularizers.L2(weight_decay)

    all_layers = []
    t_cv = layers.Conv3DTranspose(filters=filters, 
                                    kernel_size=kernel_size, 
                                    strides=strides, 
                                    padding=padding, 
                                    data_format=data_format,
                                    kernel_initializer=kernel_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_initializer=tf.constant_initializer(0.),
                                    activation=activation_fn)
    all_layers.append(t_cv)
    
    if batch_norm:
        if data_format == "channels_last":
            axis = -1
        else:
            axis = 1
        bn = layers.BatchNormalization(axis=axis, momentum=batch_norm_decay)
        all_layers.append(bn)

    return Sequential(all_layers, name=name)

def get_transpose_out_shape(in_shape, filters, kernel_size, strides, padding):
    # get output shape of a 3D transpose operation
    if padding == 'SAME':
        # according to https://github.com/keras-team/keras/blob/2c48a3b38b6b6139be2da501982fd2f61d7d48fe/keras/utils/conv_utils.py#L157
        new_rows = in_shape[1] * strides[1]
        new_cols = in_shape[2] * strides[1]
    elif padding == 'VALID':
        # according to https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
        new_rows = ((in_shape[1] - 1) * strides[1] + kernel_size[1])
        new_cols = ((in_shape[2] - 1) * strides[2] + kernel_size[2])
    return (in_shape[0], new_rows, new_cols, filters)
        

