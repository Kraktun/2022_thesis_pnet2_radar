from splashlib.nn.keras_layers import *
from splashlib.nn.utils_layers import *
from tensorflow.keras import layers

def get_encoder(in_layer, enc_args, preproc_model_builder):
    filters = enc_args.pop('filters', None)
    c1 = getConv3DPointLayer(filters=filters[0], kernel_size=[1,1,preproc_model_builder.out_features()], name="Conv1", **enc_args)(in_layer)
    if "in_dropout" in enc_args.keys():
        c1 = layers.Dropout(enc_args['in_dropout'])(c1)
    c2 = getConv3DPointLayer(filters=filters[1], kernel_size=[1,1,1], name="Conv2", **enc_args)(c1)
    if "in_dropout" in enc_args.keys():
        c2 = layers.Dropout(enc_args['in_dropout'])(c2)
    c3 = getConv3DPointLayer(filters=filters[2], kernel_size=[1,1,1], name="Conv3", **enc_args)(c2)
    if "in_dropout" in enc_args.keys():
        c3 = layers.Dropout(enc_args['in_dropout'])(c3)
    c4 = getConv3DPointLayer(filters=filters[3], kernel_size=[1,1,1], name="Conv4", **enc_args)(c3)
    if "in_dropout" in enc_args.keys():
        c4 = layers.Dropout(enc_args['in_dropout'])(c4)
    c5 = getConv3DPointLayer(filters=filters[4], kernel_size=[1,1,1], name="Conv5", **enc_args)(c4)
    if "post_dropout" in enc_args.keys():
        c5 = layers.Dropout(enc_args['post_dropout'])(c5)
    mp = layers.MaxPooling3D(pool_size=[1,preproc_model_builder.points_in_frame,1], strides=[1,2,2], padding=enc_args["padding"], data_format=enc_args["data_format"])(c5)
    res = layers.Reshape((in_layer.shape[1],-1))(mp)
    encoder = res
    return encoder

def get_encoder_fc(in_layer, enc_args, preproc_model_builder):
    filters = enc_args['filters']
    middle = get_encoder(in_layer, enc_args, preproc_model_builder)
    encoder = layers.Dense(units=filters[4]//2)(middle)
    return encoder

def get_decoder_fc(in_layer, dec_args, preproc_model_builder):
    pmb = preproc_model_builder
    units = dec_args.pop('units', None)
    # replace None and negative values with output from rnn
    units = [u if u is not None and u > 0 else in_layer.shape[2] for u in units]
    fc1 = layers.Dense(units=units[0], name='fc1', **dec_args)(in_layer)
    if "in_dropout" in dec_args.keys():
        fc1 = layers.Dropout(dec_args['in_dropout'])(fc1)
    fc2 = layers.Dense(units=units[1], name='fc2', **dec_args)(fc1)
    if "post_dropout" in dec_args.keys():
        fc2 = layers.Dropout(dec_args['post_dropout'])(fc2)
    fc3 = layers.Dense(units=pmb.points_in_frame*pmb.out_features(), name='fc3', **dec_args)(fc2)
    # same as input_layer.shape[1:-1]
    res = layers.Reshape((pmb.out_time_steps, pmb.points_in_frame, pmb.out_features()))(fc3)
    decoder = res
    return decoder

def get_decoder_upconv(in_layer, dec_args, preproc_model_builder):
    pmb = preproc_model_builder
    # get args
    fc_args = dec_args['fc_args']
    upconv_args = dec_args['upconv_args']

    # extract fc args
    units = fc_args.pop('units', None)
    # replace None and negative values with output from rnn
    units = [u if u is not None and u > 0 else in_layer.shape[2] for u in units]
    
    # extract upconv args
    max_pool_size = upconv_args.pop('max_pool_size', 4)
    max_pool_stride = upconv_args.pop('max_pool_stride', 4)
    max_pool_padding = upconv_args.pop('max_pool_padding', 'SAME')
    kernel_sizes = upconv_args.pop('kernel_sizes', [[1,2,2], [1,3,3], [1,4,4], [1,5,5]])
    strides = upconv_args.pop('strides', [[1,1,1], [1,1,1], [1,2,2], [1,2,2]])
    filters_dividers = upconv_args.pop('filters_dividers', [1, 2, 4, 8])
    
    # fc decoder
    fc1 = layers.Dense(units=units[0], name='fc1', **fc_args)(in_layer)
    if "in_dropout" in dec_args.keys():
        fc1 = layers.Dropout(dec_args['in_dropout'])(fc1)
    fc2 = layers.Dense(units=units[1], name='fc2', **fc_args)(fc1)
    if "in_dropout" in dec_args.keys():
        fc2 = layers.Dropout(dec_args['in_dropout'])(fc2)
    fc3 = layers.Dense(units=units[2], name='fc3', activation=None)(fc2)
    res = layers.Reshape((pmb.out_time_steps, -1, pmb.out_features()))(fc3)
    fc_decoder = res
    # upconv decoder
    res = layers.Reshape((pmb.out_time_steps, 1, 1, -1))(in_layer)
    in_filters = res.shape[4]
    
    if in_filters // max(filters_dividers) < 1:
        raise ValueError(f"Input shape of upconv is too small. Should be at least {max(filters_dividers)}")

    upconv1 = get_conv3D_point_transpose_layer(filters=in_filters//filters_dividers[0], 
                                                kernel_size=kernel_sizes[0], 
                                                strides=strides[0], 
                                                data_format='channels_last', 
                                                name='upconv1',
                                                **upconv_args)(res)
    upconv2 = get_conv3D_point_transpose_layer(filters=in_filters//filters_dividers[1], 
                                                kernel_size=kernel_sizes[1], 
                                                strides=strides[1], 
                                                data_format='channels_last', 
                                                name='upconv2',
                                                **upconv_args)(upconv1)
    upconv3 = get_conv3D_point_transpose_layer(filters=in_filters//filters_dividers[2], 
                                                kernel_size=kernel_sizes[2], 
                                                strides=strides[2], 
                                                data_format='channels_last', 
                                                name='upconv3',
                                                **upconv_args)(upconv2)
    upconv4 = get_conv3D_point_transpose_layer(filters=in_filters//filters_dividers[3], 
                                                kernel_size=kernel_sizes[3], 
                                                strides=strides[3], 
                                                data_format='channels_last', 
                                                name='upconv4',
                                                **upconv_args)(upconv3)
    upconv5 = get_conv3D_point_transpose_layer(filters=pmb.out_features(), 
                                                kernel_size=[1,1,1], 
                                                strides=[1,1,1], 
                                                padding='VALID', 
                                                batch_norm=False, 
                                                activation_fn=None, 
                                                data_format='channels_last', 
                                                name='upconv5')(upconv4)
    res = layers.Reshape((pmb.out_time_steps, -1, pmb.out_features()))(upconv5)
    if "in_dropout" in dec_args.keys():
        res = layers.Dropout(dec_args['in_dropout'])(res)
    mp = layers.MaxPooling2D(pool_size=[1,max_pool_size], 
                                strides=[1,max_pool_stride], 
                                padding=max_pool_padding, 
                                data_format='channels_last')(res)
    upconv_decoder = mp
    # concatenate the two outputs
    conc = layers.Concatenate(axis=2)([fc_decoder, upconv_decoder])
    # final fc layer, note that in the original this is not present, but then the model would not work
    res = layers.Reshape((conc.shape[1], conc.shape[2]*conc.shape[3]))(conc)
    if "post_dropout" in dec_args.keys():
        res = layers.Dropout(dec_args['post_dropout'])(res)
    fc_ult = layers.Dense(units=pmb.points_in_frame*pmb.out_features(), name='fc_conc')(res)
    res = layers.Reshape((pmb.out_time_steps, -1, pmb.out_features()))(fc_ult)
    decoder = res
    return decoder