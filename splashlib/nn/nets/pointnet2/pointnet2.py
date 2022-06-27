from .layers import *
from tensorflow.keras import layers as tf_layers
from tensorflow.keras.layers import Layer
from splashlib.nn.utils_layers import parse_regularizers

# In Pointnet2 with 'xyz' we refer to the points in the metric space, while with 'points' we refer to the features associated to the metric points

def _get_encoder_layers(enc_args):
    enc_layers = []
    for s_arg in enc_args['sa_settings']:
        lay = PointNetSetAbstraction(**s_arg)
        enc_layers.append(lay)
    return enc_layers

def _get_decoder_layers(dec_args):
    dec_layers = []
    for d_arg in dec_args['fp_settings']:
        lay = PointNetFeaturePropagation(**d_arg)
        dec_layers.append(lay)
    return dec_layers

class Pointnet2Encoder(Layer):

    def __init__(self, enc_args, name="Pointnet2Encoder", **kwargs):
        super(Pointnet2Encoder, self).__init__(name=name, **kwargs)
        self.enc_args = enc_args
        self.enc_layers = _get_encoder_layers(enc_args)
        self.post_dropout = "post_dropout" in enc_args.keys()
        
    def build(self, input_shape):
        if self.post_dropout:
            self.post_drop_layer = tf_layers.Dropout(self.enc_args['post_dropout'])
    
    def call(self, inputs):
        encoder_states = []
        enc_outputs = []
        for i in range(inputs.shape[1]): # = pmb.out_time_steps
            new_xyz = inputs[:,i,:,:3]
            new_points = inputs[:,i,:,3:]
            states = []
            states.append([new_xyz, new_points])
            new_p = []
            for lay in self.enc_layers:
                new_xyz, new_points = lay(new_xyz, new_points)
                states.append([new_xyz, new_points])
                new_p.append(tf.reshape(new_points, shape=(-1, new_points.shape[1]*new_points.shape[2],)))
            encoder_states.append(states)
            enc_outputs.append(tf.stack(new_p, axis=1))
        encoder = tf.stack(enc_outputs, axis=1)
        if self.post_dropout:
            encoder = self.post_drop_layer(encoder)
        return encoder_states, encoder

    def get_config(self):
        config = {
            'enc_args': self.enc_args, 
        }
        base_config = super(Pointnet2Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Pointnet2Decoder(Layer):

    def __init__(self, dec_args, name="Pointnet2Decoder", **kwargs):
        super(Pointnet2Decoder, self).__init__(name=name, **kwargs)
        self.dec_args = dec_args
        self.dec_layers = _get_decoder_layers(dec_args)
        self.append_xyz = 'append_xyz' in dec_args.keys() and dec_args['append_xyz']
        self.pre_dropout = "pre_dropout" in dec_args.keys()
        self.post_dropout = "post_dropout" in dec_args.keys()
        self.replace_inf = dec_args['replace_inf']
        self.net_args = []
        if "net_settings" in dec_args.keys():
            self.net_args = dec_args['net_settings']
    
    def build(self, input_shape):
        rnn_shape, encoder_states_shape = input_shape
        # rnn_shape = (None, time_steps, num_SA_layers, last_enc_points*last_enc_filters)
        last_encoder_timestep_shape = encoder_states_shape[-1] # all SA outputs of last time step
        last_enc_points_shape = last_encoder_timestep_shape[-1][0] # xyz points, shape is (None, last_enc_points, 3)
        last_enc_feats_shape = last_encoder_timestep_shape[-1][1] # features, shape is (None, last_enc_points, last_enc_filters)
        first_in_points = encoder_states_shape[-1][0][0] # input to first SA layer (last time step), shape is (None, number_of_in_points, 3)
        
        self.time_steps = rnn_shape[1]
        self.in_points = first_in_points[-2]
        if "out_points" not in self.dec_args.keys() or self.dec_args['out_points'] < 0:
            self.out_points = last_enc_points_shape[-2]
        else:
            self.out_points = self.dec_args['out_points']
        self.enc_in_points = last_enc_points_shape[-2]
        self.enc_in_points_dim = last_enc_points_shape[-1]
        self.enc_in_feats = last_enc_feats_shape[-2]

        self.reshape_1 = tf_layers.Reshape((self.time_steps, self.enc_in_points*self.enc_in_points_dim))
        self.spatial_dense = tf_layers.Dense(self.in_points)
        self.concat = tf_layers.Concatenate(axis=-1)

        # user defined layers
        self.net_layers = []
        for lay in self.net_args:
            net_type = lay["type"]
            if "args" in lay.keys():
                net_args = lay["args"]
            else:
                net_args = {}
            if net_type.lower() == "conv2d":
                self.net_layers.append(tf_layers.Conv2D(**net_args))
            elif net_type.lower() == "maxpooling2d":
                self.net_layers.append(tf_layers.MaxPooling2D(**net_args))
            elif net_type.lower() == "gru":
                net_args = parse_regularizers(net_args)
                self.net_layers.append(tf_layers.GRU(**net_args))
                if "return_sequences" not in net_args.keys() or not net_args['return_sequences']:
                    self.time_steps = 1
            elif net_type.lower() == "batchnormalization":
                self.net_layers.append(tf_layers.BatchNormalization(**net_args))
            elif net_type.lower() == "dense":
                self.net_layers.append(tf_layers.Dense(**net_args))
            elif net_type.lower() == "reshape":
                target_shape = net_args['target_shape']
                if target_shape == -1:
                    target_shape = (self.time_steps, -1)
                self.net_layers.append(tf_layers.Reshape(target_shape=target_shape))
            
        self.reshape_2 = tf_layers.Reshape((self.time_steps, -1))
        self.last_dense = tf_layers.Dense(self.out_points*self.enc_in_points_dim)
        self.reshape_3 = tf_layers.Reshape((self.time_steps, self.out_points, self.enc_in_points_dim))
        if self.pre_dropout:
            self.pre_drop_layer = tf_layers.Dropout(self.dec_args['pre_dropout'])
        if self.post_dropout:
            self.post_drop_layer = tf_layers.Dropout(self.dec_args['post_dropout'])

    def call(self, inputs):
        rnn, encoder_states = inputs
        # add dropout if set
        if self.pre_dropout:
            rnn = self.pre_drop_layer(rnn)
        rnn = tf.reverse(rnn, axis=[-2])
        dec_outputs = []
        if len(self.dec_layers) > 0:
            for ts, states in enumerate(encoder_states): # loop time steps
                states.reverse() # reverse, last to first SA layers
                for i, dec_layer in enumerate(self.dec_layers):
                    prev_xyz = states[i][0]
                    curr_xyz = states[i+1][0]
                    # repeat first point and reuse when source is inf, alternatively it could be set to zero
                    if self.replace_inf == "first":
                        first_rep = tf.expand_dims(curr_xyz[:,0,:], 1)
                        first_rep = tf.tile(first_rep, [1,curr_xyz.shape[-2],1])
                        curr_xyz = tf.where(tf.math.is_finite(curr_xyz), curr_xyz, first_rep) 
                    elif self.replace_inf == "zero":
                        zero_rep = tf.tile([[[0.0]]], [1,curr_xyz.shape[-2],curr_xyz.shape[-1]])
                        curr_xyz = tf.where(tf.math.is_finite(curr_xyz), curr_xyz, zero_rep) 
                    if i == 0:
                        prev_points = tf.reshape(rnn[:,ts,i,:], shape=(-1, states[i][1].shape[1], states[i][1].shape[2]))
                    if i == len(self.dec_layers)-1:
                        curr_points = None
                    else:
                        curr_points = tf.reshape(rnn[:,ts,i+1,:], shape=(-1, states[i+1][1].shape[1], states[i+1][1].shape[2]))
                    curr_points = dec_layer(curr_xyz, prev_xyz, curr_points, prev_points)
                    prev_points = curr_points
                dec_outputs.append(curr_points)
            decoder = tf.stack(dec_outputs, axis=1)
        else:
            decoder = rnn
            
        
        if self.append_xyz:
            # add information from xyz points of the last SA layer
            encoded_points = [st[0][0] for st in encoder_states] # get xyz of last SA layer, remember it's reversed (it's the first index of the two)
            encoded_points = tf.stack(encoded_points, axis=1) # stack, shape is (None, time_steps, num_last_sa_points, 3)
            encoded_points = self.reshape_1(encoded_points) # shape is (None, time_steps, num_last_sa_points*3)
            encoded_points = self.spatial_dense(encoded_points) # shape is (None, time_steps, num_in_points)
            encoded_points = tf.expand_dims(encoded_points, axis=-1) # shape is (None, time_steps, num_in_points, 1)
            decoder = self.concat([decoder, encoded_points]) # shape is (None, time_steps, num_in_points, num_last_sa_feats + 1)
        # add dropout if set
        if self.post_dropout:
            decoder = self.post_drop_layer(decoder)
        
        for lay in self.net_layers:
            decoder = lay(decoder)
        
        decoder = self.reshape_2(decoder) # shape is (None, time_steps, ceil(num_in_points/32)*128)
        decoder = self.last_dense(decoder) # shape is (None, time_steps, num_out_points*3)
        decoder = self.reshape_3(decoder) # shape is (None, time_steps, num_out_points, 3)
        
        return decoder, dec_outputs

    def get_config(self):
        config = {
            'dec_args': self.dec_args, 
        }
        base_config = super(Pointnet2Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Pointnet2DecoderCore(Layer):

    def __init__(self, dec_args, name="Pointnet2DecoderCore", **kwargs):
        super(Pointnet2DecoderCore, self).__init__(name=name, **kwargs)
        self.dec_args = dec_args
        self.dec_layers = _get_decoder_layers(dec_args)
        self.append_xyz = 'append_xyz' in dec_args.keys() and dec_args['append_xyz']
        self.pre_dropout = "pre_dropout" in dec_args.keys()
        self.post_dropout = "post_dropout" in dec_args.keys()
        self.replace_inf = dec_args['replace_inf']
    
    def build(self, input_shape):
        rnn_shape, encoder_states_shape = input_shape
        # rnn_shape = (None, time_steps, num_SA_layers, last_enc_points*last_enc_filters)
        last_encoder_timestep_shape = encoder_states_shape[-1] # all SA outputs of last time step
        last_enc_points_shape = last_encoder_timestep_shape[-1][0] # xyz points, shape is (None, last_enc_points, 3)
        last_enc_feats_shape = last_encoder_timestep_shape[-1][1] # features, shape is (None, last_enc_points, last_enc_filters)
        first_in_points = encoder_states_shape[-1][0][0] # input to first SA layer (last time step), shape is (None, number_of_in_points, 3)
        
        self.time_steps = rnn_shape[1]
        self.in_points = first_in_points[-2]
        if "out_points" not in self.dec_args.keys() or self.dec_args['out_points'] < 0:
            self.out_points = last_enc_points_shape[-2]
        else:
            self.out_points = self.dec_args['out_points']
        self.enc_in_points = last_enc_points_shape[-2]
        self.enc_in_points_dim = last_enc_points_shape[-1]
        self.enc_in_feats = last_enc_feats_shape[-2]

        self.reshape_1 = tf_layers.Reshape((self.time_steps, self.enc_in_points*self.enc_in_points_dim))
        self.spatial_dense = tf_layers.Dense(self.in_points)
        self.concat = tf_layers.Concatenate(axis=-1)
   
        if self.pre_dropout:
            self.pre_drop_layer = tf_layers.Dropout(self.dec_args['pre_dropout'])
        if self.post_dropout:
            self.post_drop_layer = tf_layers.Dropout(self.dec_args['post_dropout'])

    def call(self, inputs):
        rnn, encoder_states = inputs
        # add dropout if set
        if self.pre_dropout:
            rnn = self.pre_drop_layer(rnn)
        rnn = tf.reverse(rnn, axis=[-2])
        dec_outputs = []
        if len(self.dec_layers) > 0:
            for ts, states in enumerate(encoder_states): # loop time steps
                states.reverse() # reverse, last to first SA layers
                for i, dec_layer in enumerate(self.dec_layers):
                    prev_xyz = states[i][0] # SA layer t, i.e. the centroids of curr_xyz
                    curr_xyz = states[i+1][0] # SA layer t-1 (the list is reversed), i.e. the bigger one
                    # repeat first point and reuse when source is inf, alternatively it could be set to zero
                    if self.replace_inf == "first":
                        first_rep = tf.expand_dims(curr_xyz[:,0,:], 1)
                        first_rep = tf.tile(first_rep, [1,curr_xyz.shape[-2],1])
                        curr_xyz = tf.where(tf.math.is_finite(curr_xyz), curr_xyz, first_rep) 
                    elif self.replace_inf == "zero":
                        zero_rep = tf.tile([[[0.0]]], [1,curr_xyz.shape[-2],curr_xyz.shape[-1]])
                        curr_xyz = tf.where(tf.math.is_finite(curr_xyz), curr_xyz, zero_rep) 
                    if i == 0:
                        prev_points = tf.reshape(rnn[:,ts,i,:], shape=(-1, states[i][1].shape[1], states[i][1].shape[2]))
                    if i == len(self.dec_layers)-1:
                        curr_points = None
                    else:
                        curr_points = tf.reshape(rnn[:,ts,i+1,:], shape=(-1, states[i+1][1].shape[1], states[i+1][1].shape[2]))
                    curr_points = dec_layer(curr_xyz, prev_xyz, curr_points, prev_points)
                    prev_points = curr_points
                dec_outputs.append(curr_points)
            decoder = tf.stack(dec_outputs, axis=1)
        else:
            decoder = rnn
        
        if self.append_xyz:
            # add information from xyz points of the last SA layer
            encoded_points = [st[0][0] for st in encoder_states] # get xyz of last SA layer, remember it's reversed (it's the first index of the two)
            encoded_points = tf.stack(encoded_points, axis=1) # stack, shape is (None, time_steps, num_last_sa_points, 3)
            encoded_points = self.reshape_1(encoded_points) # shape is (None, time_steps, num_last_sa_points*3)
            encoded_points = self.spatial_dense(encoded_points) # shape is (None, time_steps, num_in_points)
            encoded_points = tf.expand_dims(encoded_points, axis=-1) # shape is (None, time_steps, num_in_points, 1)
            decoder = self.concat([decoder, encoded_points]) # shape is (None, time_steps, num_in_points, num_last_sa_feats + 1)
        # add dropout if set
        if self.post_dropout:
            decoder = self.post_drop_layer(decoder)
        return decoder, dec_outputs

    def get_config(self):
        config = {
            'dec_args': self.dec_args, 
        }
        base_config = super(Pointnet2DecoderCore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DecoderExtension(Layer):

    def __init__(self, dec_args, mode, name="DecoderExtension", **kwargs):
        super(DecoderExtension, self).__init__(name=name, **kwargs)
        self.dec_args = dec_args
        self.net_args = []
        self.mode = mode
        if mode == "supervised" and "model_dec_supervised" in dec_args.keys():
            self.net_args = dec_args["model_dec_supervised"]['net_settings']
        elif mode == "unsupervised" and "model_dec_supervised" in dec_args.keys():
            self.net_args = dec_args["model_dec_unsupervised"]['net_settings']
    
    def build(self, input_shape):
        decoder_shape = input_shape
        # decoder_shape = (None, time_steps, num_in_points, num_feats)
        self.time_steps = decoder_shape[-3]
        self.in_points = decoder_shape[-2]
        if "out_points" not in self.dec_args.keys() or self.dec_args['out_points'] < 0:
            self.out_points = self.in_points
        else:
            self.out_points = self.dec_args['out_points']
        if self.mode == "unsupervised":
            if self.dec_args["model_dec_unsupervised"]['target'] == "x":
                self.out_points = self.in_points
        self.enc_in_points_dim = 3 # need only xyz

        # user defined layers
        self.net_layers = []
        for lay in self.net_args:
            net_type = lay["type"]
            if "args" in lay.keys():
                net_args = lay["args"]
            else:
                net_args = {}
            if net_type.lower() == "conv2d":
                self.net_layers.append(tf_layers.Conv2D(**net_args))
            elif net_type.lower() == "maxpooling2d":
                self.net_layers.append(tf_layers.MaxPooling2D(**net_args))
            elif net_type.lower() == "gru":
                net_args = parse_regularizers(net_args)
                self.net_layers.append(tf_layers.GRU(**net_args))
                if "return_sequences" not in net_args.keys() or not net_args['return_sequences']:
                    self.time_steps = 1
            elif net_type.lower() == "batchnormalization":
                self.net_layers.append(tf_layers.BatchNormalization(**net_args))
            elif net_type.lower() == "dense":
                self.net_layers.append(tf_layers.Dense(**net_args))
            elif net_type.lower() == "reshape":
                target_shape = net_args['target_shape']
                if target_shape == -1:
                    target_shape = (self.time_steps, -1)
                self.net_layers.append(tf_layers.Reshape(target_shape=target_shape))
            
        self.reshape_2 = tf_layers.Reshape((self.time_steps, -1))
        self.last_dense = tf_layers.Dense(self.out_points*self.enc_in_points_dim)
        self.reshape_3 = tf_layers.Reshape((self.time_steps, self.out_points, self.enc_in_points_dim))

    def call(self, inputs):
        # shape is (None, time_steps, num_in_points, num_feats)
        decoder = inputs
        for lay in self.net_layers:
            decoder = lay(decoder)
        decoder = self.reshape_2(decoder) # shape is (None, time_steps, ceil(num_in_points/32)*128)
        decoder = self.last_dense(decoder) # shape is (None, time_steps, num_out_points*3)
        decoder = self.reshape_3(decoder) # shape is (None, time_steps, num_out_points, 3)
        
        return decoder

    def get_config(self):
        config = {
            'dec_args': self.dec_args, 
        }
        base_config = super(DecoderExtension, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
