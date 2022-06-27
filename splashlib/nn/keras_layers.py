import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import layers

from splashlib.nn.utils_layers import parse_regularizers

"""
Collection of general layers used for the models.
"""

class CnnSequence(Layer):

    def __init__(self, rnn_type, rnn_args, name="CnnSequence", **kwargs):
        super(CnnSequence, self).__init__(name=name, **kwargs)
        self.rnn_type = rnn_type
        self.rnn_args = rnn_args

    def build(self, input_shape):
        encoder_shape, states_shape = input_shape
        # encoder_shape = (None, time_steps, num_SA_layers, last_enc_points*last_enc_filters)
        # last_enc_points = centroids
        
        last_encoder_timestep_shape = states_shape[-1] # all SA outputs of last time step
        last_enc_feats_shape = last_encoder_timestep_shape[-1][1] # features, shape is (None, last_enc_points, last_enc_filters)
        time_steps = encoder_shape[1]
        
        self.init_reshape = layers.Reshape(target_shape=(time_steps, -1, last_enc_feats_shape[-2], last_enc_feats_shape[-1]))
        
        pooled_feat = last_enc_feats_shape[-2]
        last_cnn_filters = last_enc_feats_shape[-1]
        self.net_layers = []
        for lay in self.rnn_args['net_args']:
            net_type = lay["type"]
            if "args" in lay.keys():
                net_args = lay["args"]
            else:
                net_args = {}
            if net_type.lower() == "conv2d":
                self.net_layers.append(layers.Conv2D(**net_args))
                last_cnn_filters = net_args['filters']
            elif net_type.lower() == "maxpooling2d":
                self.net_layers.append(layers.MaxPooling2D(**net_args))
                pooled_feat = pooled_feat // net_args['pool_size'][-1]
            elif net_type.lower() == "maxpooling3d":
                self.net_layers.append(layers.MaxPooling3D(**net_args))
                pooled_feat = pooled_feat // net_args['pool_size'][-1]
            elif net_type.lower() == "batchnormalization":
                self.net_layers.append(layers.BatchNormalization(**net_args))
            elif net_type.lower() == "dense":
                self.net_layers.append(layers.Dense(**net_args))
            elif net_type.lower() == "reshape":
                self.net_layers.append(layers.Reshape(**net_args))
        
        # assume padding="SAME" or it's a mess to compute
        self.out_reshape = layers.Reshape(target_shape=(1, -1, pooled_feat*last_cnn_filters))
        
        self.input_transform = self.rnn_args['input_transform']
        self.output_transform = self.rnn_args['output_transform']
        self.unstack_layer = layers.Reshape(target_shape=(encoder_shape[-3], encoder_shape[-2]*encoder_shape[-1]))
        self.return_sequences = self.rnn_args['return_sequences']

    def call(self, inputs):
        out, encoder_states = inputs
        # out.shape is (None, time_steps, num_SA_layers, num_last_filters*centroids) e.g. (None, 30, 3, 256)

        unstack_axis = -2
        transpose_order = [0,2,1,3]
        if "init_reshape" in self.input_transform:
            out = self.init_reshape(out)
            unstack_axis = -3
            transpose_order = [0,2,1,3,4]
        if "unstack" in self.input_transform:
            out = tf.unstack(out, axis=unstack_axis)
        if "layers_first" in self.input_transform:
            out = tf.transpose(out, transpose_order)
        if self.rnn_type == 'cnn_sequence':
            for lay in self.net_layers:
                out = lay(out)
        # restore original shape
        if "layers_first" in self.input_transform:
            out = tf.transpose(out, transpose_order)
        if not self.return_sequences:
            encoder_states = [encoder_states[-1]]
            out = out[:,-1]
        if "out_reshape" in self.output_transform:
            out = self.out_reshape(out)
        if "stack" in self.output_transform:
            out = tf.stack(out, axis=-2)
        return out, encoder_states

    def get_config(self):
        config = {
            'rnn_args': self.rnn_args, 
            'rnn_type': self.rnn_type
        }
        base_config = super(CnnSequence, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class IdentityLayer(Layer):

    def __init__(self, rnn_type, rnn_args, name="IdentityLayer", **kwargs):
        super(IdentityLayer, self).__init__(name=name, **kwargs)
        self.rnn_type = rnn_type
        self.rnn_args = rnn_args

    def call(self, inputs):
        if self.rnn_type == 'identity':
            return inputs
        elif self.rnn_type == 'identity_last_frame':
            rnn = inputs[:,-1]
            rnn = tf.expand_dims(rnn, 1)
        return rnn

    def get_config(self):
        config = {
            'rnn_args': self.rnn_args, 
            'rnn_type': self.rnn_type
        }
        base_config = super(IdentityLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConcatRnn(Layer):
    # wrapper around a rnn that first concatenates the inputs then apply the rnn and then splits them
    # i.e. input has shape (None, time_steps, num_filters, feats)
    # rnn needs input with 3 dims, so concatenate the input in a single vector (None, time_steps, num_filters*feats)
    # and call the rnn 
    # then reshape to the original shape (None, time_steps, num_filters, feats)
    # this layer currently requires that the rnn has a number of units equal to num_filters*feats

    def __init__(self, rnn_type, rnn_args, name="ConcatRnn", **kwargs):
        super(ConcatRnn, self).__init__(name=name, **kwargs)
        self.rnn_type = rnn_type
        self.rnn_args = rnn_args
        
    def build(self, input_shape):
        # input shape is (None, time_steps, num_SA_layers, filters)
        # overwrite # of units or the reshape fails
        my_args = self.rnn_args.copy()
        if "units" in my_args.keys() and my_args["units"] < 0:
            my_args["units"] = input_shape[-2]*input_shape[-1]
        if 'bidirectional' in my_args:
            bidirectional = my_args.pop('bidirectional')
        else:
            bidirectional = False
        
        if bidirectional:
            my_args["units"] = my_args["units"]//2
            if self.rnn_type == 'gru':
                self.rnn_layer = layers.Bidirectional(layers.GRU(name='gru', **my_args))
            elif self.rnn_type == 'lstm':
                self.rnn_layer = layers.Bidirectional(layers.LSTM(name='lstm', **my_args))
        else:
            if self.rnn_type == 'gru':
                self.rnn_layer = layers.GRU(name='gru', **my_args)
            elif self.rnn_type == 'lstm':
                self.rnn_layer = layers.LSTM(name='lstm', **my_args)
    
        self.reshape_layer = layers.Reshape(target_shape=(input_shape[-3], input_shape[-2]*input_shape[-1]))
        if my_args['return_sequences']:
            out_sequences = input_shape[-3]
        else:
            out_sequences = 1
        if my_args["units"] == input_shape[-2]*input_shape[-1] or (bidirectional and my_args["units"] == input_shape[-2]*input_shape[-1]//2):
            self.restore_shape = layers.Reshape(target_shape=(out_sequences, input_shape[-2], input_shape[-1]))
        else:
            self.restore_shape = layers.Reshape(target_shape=(out_sequences, input_shape[-2], -1))
    
    def call(self, inputs):
        enc_vector = self.reshape_layer(inputs)
        rnn = self.rnn_layer(enc_vector)
        rnn = self.restore_shape(rnn)
        return rnn

    def get_config(self):
        config = {
            'rnn_args': self.rnn_args, 
            'rnn_type': self.rnn_type
        }
        base_config = super(ConcatRnn, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class UnstackedRnn(Layer):
    # wrapper around a rnn that first unstack the filters then apply the rnn and restack them
    # i.e. input has shape (None, time_steps, num_filters, feats)
    # rnn needs input with 3 dims, so divide the input in a list of filters [(None, time_steps, feats)]
    # and call the rnn with each one, one at a time (from the first to the last)
    # then restack the results and get in output (None, time_steps, num_filters, new_feats)
    # new_feats depends on the number of units of the rnn

    def __init__(self, rnn_type, rnn_args, name="UnstackedRnn", **kwargs):
        super(UnstackedRnn, self).__init__(name=name, **kwargs)
        self.rnn_type = rnn_type
        self.rnn_args = rnn_args
    
    def build(self, input_shape):
        num_filters = input_shape[-2]
        my_args = self.rnn_args.copy()

        if 'rnn_mode' in my_args:
            self.rnn_mode = my_args.pop('rnn_mode')
        else:
            self.rnn_mode = "same"
        
        if 'bidirectional' in my_args:
            bidirectional = my_args.pop('bidirectional')
        else:
            bidirectional = False

        if "units" in my_args.keys() and my_args["units"] < 0:
            my_args["units"] = input_shape[-1]
        
        if self.rnn_mode == "same":
            if self.rnn_type == 'gru':
                if bidirectional:
                    my_args["units"] = my_args["units"]//2
                    self.rnn_layer = layers.Bidirectional(layers.GRU(name='gru', **my_args))
                else:
                    self.rnn_layer = layers.GRU(name='gru', **my_args)
            elif self.rnn_type == 'lstm':
                if bidirectional:
                    my_args["units"] = my_args["units"]//2
                    self.rnn_layer = layers.Bidirectional(layers.LSTM(name='lstm', **my_args))
                else:
                    self.rnn_layer = layers.LSTM(name='lstm', **my_args)
        if self.rnn_mode == "double_same":
            # allow different number of units
            my_args0 = my_args['rnn0']
            my_args0 = parse_regularizers(my_args0)
            my_args1 = my_args['rnn1']
            my_args1 = parse_regularizers(my_args1)
            if my_args1["units"] < 0:
                my_args1["units"] = input_shape[-1]
            if self.rnn_type == 'gru':
                if bidirectional:
                    my_args0["units"] = my_args0["units"]//2
                    my_args1["units"] = my_args1["units"]//2
                    self.rnn_layers = [layers.Bidirectional(layers.GRU(name='gru0', **my_args0)), layers.Bidirectional(layers.GRU(name='gru1', **my_args1))]
                else:
                    self.rnn_layers = [layers.GRU(name='gru0', **my_args0), layers.GRU(name='gru1', **my_args1)]
            elif self.rnn_type == 'lstm':
                if bidirectional:
                    self.rnn_layers = [layers.Bidirectional(layers.LSTM(name='lstm0', **my_args0)), layers.Bidirectional(layers.LSTM(name='lstm1', **my_args1))]
                else:
                    self.rnn_layers = [layers.LSTM(name='lstm0', **my_args0), layers.LSTM(name='lstm1', **my_args1)]
        elif self.rnn_mode == "parallel":
            self.rnn_layers = []
            for i in range(num_filters):
                if self.rnn_type == 'gru':
                    if bidirectional:
                        my_args["units"] = my_args["units"]//2
                        self.rnn_layers.append(layers.Bidirectional(layers.GRU(name=f'gru_{i}', **my_args)))
                    else:
                        self.rnn_layers.append(layers.GRU(name=f'gru_{i}', **my_args))
                elif self.rnn_type == 'lstm':
                    if bidirectional:
                        my_args["units"] = my_args["units"]//2
                        self.rnn_layers.append(layers.Bidirectional(layers.LSTM(name=f'lstm_{i}', **my_args)))
                    else:
                        self.rnn_layers.append(layers.LSTM(name=f'lstm_{i}', **my_args))
        elif self.rnn_mode == "double_parallel":
            my_args0 = my_args['rnn0']
            my_args0 = parse_regularizers(my_args0)
            my_args1 = my_args['rnn1']
            my_args1 = parse_regularizers(my_args1)
            if my_args1["units"] < 0:
                my_args1["units"] = input_shape[-1]
            self.rnn_layers = []
            for i in range(num_filters):
                if self.rnn_type == 'gru':
                    if bidirectional:
                        my_args0["units"] = my_args0["units"]//2
                        my_args1["units"] = my_args1["units"]//2
                        self.rnn_layers.append([layers.Bidirectional(layers.GRU(name=f'gru0_{i}', **my_args0)), layers.Bidirectional(layers.GRU(name=f'gru1_{i}', **my_args1))])
                    else:
                        self.rnn_layers.append([layers.GRU(name=f'gru0_{i}', **my_args0), layers.GRU(name=f'gru1_{i}', **my_args1)])
                elif self.rnn_type == 'lstm':
                    if bidirectional:
                        self.rnn_layers.append([layers.Bidirectional(layers.LSTM(name=f'lstm0_{i}', **my_args0)), layers.Bidirectional(layers.LSTM(name=f'lstm0_{i}', **my_args1))])
                    else:
                        self.rnn_layers.append([layers.LSTM(name=f'lstm0_{i}', **my_args0), layers.LSTM(name=f'lstm0_{i}', **my_args1)])

    def call(self, inputs):
        rnn = []
        encoded_filters = tf.unstack(inputs, axis=-2)
        if self.rnn_mode == "same":
            for enc_points in encoded_filters:
                # reuse the same rnn for all SA layers
                rnn.append(self.rnn_layer(enc_points))
        elif self.rnn_mode == "double_same":
            for enc_points in encoded_filters:
                out_1 = self.rnn_layers[0](enc_points)
                rnn.append(self.rnn_layers[1](out_1))
        elif self.rnn_mode == "parallel":
            for i, enc_points in enumerate(encoded_filters):
                rnn.append(self.rnn_layers[i](enc_points))
        elif self.rnn_mode == "double_parallel":
            for i, enc_points in enumerate(encoded_filters):
                out_1 = self.rnn_layers[i][0](enc_points)
                rnn.append(self.rnn_layers[i][1](out_1))
        rnn = tf.stack(rnn, axis=-2)
        if not self.rnn_args['return_sequences']:
            rnn = tf.expand_dims(rnn, axis=1)
        return rnn

    def get_config(self):
        config = {
            'rnn_args': self.rnn_args, 
            'rnn_type': self.rnn_type
        }
        base_config = super(UnstackedRnn, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RandomRotation(Layer):

    # shape must be (batch_size, time_steps, points_per_frame, feat_dims)
    # rotate the points in each frame (rotation is contant in a batch) around the origin (0,0,0)
    # for the rotation to make sense each frame should be centered on its own centroid (or at least the previous)
    # hardcoded to use only the first three dimensions (if there are more, they are copied with no modification)
    # note that the angle intervals are multiplied by 2*pi, so an interval [-1,1] corresponds to [-2*pi, 2*pi]

    def __init__(self, prob=0.5, ang_intervals=[[-1,1]]*3, name="RandomRotation", **kwargs):
        self.prob = prob
        self.ang_intervals = ang_intervals
        super(RandomRotation, self).__init__(name=name, **kwargs)
        self.trainable = False

    def call(self, inputs, training=None):
        if not training:
            return inputs

        if np.random.random() <= self.prob:
            angle_x = np.random.random()*(self.ang_intervals[0][1]-self.ang_intervals[0][0]) + self.ang_intervals[0][0]
            angle_x = angle_x*np.pi*2
            angle_y = np.random.random()*(self.ang_intervals[1][1]-self.ang_intervals[1][0]) + self.ang_intervals[1][0]
            angle_y = angle_y*np.pi*2
            angle_z = np.random.random()*(self.ang_intervals[2][1]-self.ang_intervals[2][0]) + self.ang_intervals[2][0]
            angle_z = angle_z*np.pi*2

            sin_x = np.sin(angle_x)
            cos_x = np.cos(angle_x)
            sin_y = np.sin(angle_y)
            cos_y = np.cos(angle_y)
            sin_z = np.sin(angle_z)
            cos_z = np.cos(angle_z)
            t_x = inputs[...,0]
            t_y = inputs[...,1]
            t_z = inputs[...,2]
            if inputs.shape[-1] > 3:
                t_other = inputs[...,3:]
            # rotate xy
            t_x = t_x*cos_x - t_y*sin_x
            t_y = t_x*sin_x + t_y*cos_x
            # rotate yz
            t_y = t_y*cos_y - t_z*sin_y
            t_z = t_y*sin_y + t_z*cos_y
            # rotate xz
            t_x = t_x*cos_z - t_z*sin_z
            t_z = t_x*sin_z + t_z*cos_z
            t_all = tf.stack([t_x, t_y, t_z], axis=-1)
            if inputs.shape[-1] > 3:
                t_all = tf.concat([t_all, t_other], axis=-1)
            return t_all
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return input_shape

    def get_config(self):
        config = {
            'prob': self.prob,
            'ang_intervals': self.ang_intervals
        }
        base_config = super(RandomRotation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RandomCloudPermutation(Layer):

    # shape must be (batch_size, time_steps, points_per_frame, feat_dims)
    # shuffle the points in each frame (shuffle is contant in a batch)

    def __init__(self, prob=0.5, name="RandomCloudPermutation", **kwargs):
        self.prob = prob
        super(RandomCloudPermutation, self).__init__(name=name, **kwargs)
        self.trainable = False

    def call(self, inputs, training=None):
        if not training:
            return inputs
        if np.random.random() <= self.prob:
            idx = np.random.permutation(inputs.shape[-2])
            return tf.gather(inputs, idx, axis=-2)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return input_shape

    def get_config(self):
        config = {
            'prob': self.prob
        }
        base_config = super(RandomCloudPermutation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RandomPermutation(Layer):

    # shape must be (batch_size, time_steps, points_per_frame, feat_dims)
    # permute the dimensions of the last axis 
    # axis = [0,1,2] means permute the coordinates of each point in the three coordinates (so not the velocity), 
    # for instance (the order of the permutation is random) (1,2,4,velocity) -> (2,4,1,velocity) 

    def __init__(self, axis=[0,1,2], prob=0.5, name="RandomPermutation", **kwargs):
        self.axis = axis
        self.prob = prob
        super(RandomPermutation, self).__init__(name=name, **kwargs)
        self.trainable = False

    def call(self, inputs, training=None):
        if not training:
            return inputs
        if np.random.random() <= self.prob:
            idx = np.random.permutation(self.axis)
            final_idx = []
            last_used_idx = 0
            for i in range(inputs.shape[-1]):
                if i in idx:
                    final_idx.append(idx[last_used_idx])
                    last_used_idx = last_used_idx+1
                # add the axis that were not permuted in the correct position
                else:
                    final_idx.append(i)
            return tf.gather(inputs, final_idx, axis=-1)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'prob': self.prob
        }
        base_config = super(RandomPermutation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class RandomShift(tf.keras.layers.Layer):

    # shape must be (batch_size, time_step, time_data)
    # adds a random shift of the points to all axis. 
    # Shift is independent for each dimension of each point.

    def __init__(self, prob=0.5, interval=(-1,1), name="RandomShift", **kwargs):
        self.prob = prob
        self.interval = interval
        super(RandomShift, self).__init__(name=name, **kwargs)
        self.trainable = False

    def call(self, inputs, training=None):
        if not training:
            return inputs
        if np.random.random() <= self.prob:
            shifts = tf.random.uniform(shape=inputs.shape[1:], minval=self.interval[0], maxval=self.interval[1])
            return tf.math.add(inputs, shifts)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return input_shape

    def get_config(self):
        config = {
            'interval': self.interval,
            'prob': self.prob
        }
        base_config = super(RandomShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

"""
DO NOT USE THE FOLLOWING LAYERS UNLESS YOU KNOW WHAT YOU ARE DOING.
They may not do what they are supposed to.
"""

class CenterSequence(tf.keras.layers.Layer):

    # shape must be (batch_size, num_frames, num_points, feats)
    # center (i.e. shift) a sequence of frames according to the center_mode provided 
    # so that the centroid of the specified frame (first, last, middle or average) is at 0
    # use_dims controls which feature axis to use to compute the centroid, if none all feat axis will be used
    # Use the map_functions on the dataset rather than this one, or your markers will not be shifted.
    # Besides, this hasn't been tested properly.

    def __init__(self, center_mode="all", use_dims=None, name="CenterSequence", **kwargs):
        super(CenterSequence, self).__init__(name=name, **kwargs)
        self.center_mode = center_mode
        self.use_dims = use_dims

    def call(self, inputs):
        feat_centers = []
        if self.use_dims is None:
            dims_arr = range(inputs.shape[-1])
        else:
            dims_arr = self.use_dims
        if self.center_mode == "all":
            for dim in dims_arr:
                centroid = tf.math.reduce_sum(inputs[...,dim], axis=-1, keepdims=True) # reduce for all points in a frame
                centroid = tf.math.reduce_sum(centroid, axis=-2, keepdims=True) # reduce for all frames in a sequence
                centroid = centroid / (inputs.shape[-2]*inputs.shape[-3]) # divide by all number of points to get the mean
                feat_centers.append(centroid)
        else:
            if self.center_mode == "first":
                take_frame = 0
            elif self.center_mode == "middle":
                take_frame = inputs.shape[-3]//2
            elif self.center_mode == "last":
                take_frame = -1
            else:
                raise ValueError("Invalid center mode in CenterSequence")
            for dim in dims_arr:
                centroid = tf.math.reduce_sum(inputs[...,dim], axis=-1, keepdims=True) # reduce for all points in a frame
                centroid = centroid[:,take_frame] # take chosen frame
                centroid = tf.expand_dims(centroid,-2) # add axis
                centroid = centroid / inputs.shape[-2] # divide only by number of points
                feat_centers.append(centroid)

        centroids = tf.stack(feat_centers, axis=-1)
        centroids = tf.tile(centroids, [1,inputs.shape[-3],inputs.shape[-2],1])
        # add 0 feat columns so that centroids matches the shape of inputs
        last_centroid_used = -1
        if self.use_dims is None:
            final_result = inputs - centroids
        else:
            for i in range(inputs.shape[-1]):
                if i == 0:
                    if i in dims_arr:
                        final_result = inputs[...,0] - centroids[...,0]
                        last_centroid_used = last_centroid_used+1
                    else:
                        final_result = inputs[...,0]
                    final_result = tf.expand_dims(final_result, -1)
                else:
                    if i in dims_arr:
                        last_centroid_used = last_centroid_used+1
                        column_to_add = inputs[...,i]-centroids[...,last_centroid_used]
                    else:
                        column_to_add = inputs[...,i]
                    column_to_add = tf.expand_dims(column_to_add, -1)
                    final_result = tf.concat([final_result, column_to_add], axis=-1)
        return final_result

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return input_shape

    def get_config(self):
        config = { }
        base_config = super(CenterSequence, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SplitLayer(Layer):

    # shape must be (batch_size, num_frames, num_features)
    # out shape is (batch_size, num_blocks, block_length, num_features)
    # Simply splits a complete sequence (i.e. without time steps) in the provided number of blocks.
    # block_size here is the same as time_steps for the dataset.

    def __init__(self, block_size=30, stride=30, name="SplitLayer", **kwargs):
        self.block_size = block_size
        self.stride = stride
        super(SplitLayer, self).__init__(name=name, **kwargs)
        self.trainable = False

    def call(self, inputs):
        block_size = int(self.block_size)
        stride = int(self.stride)
        # get total # chunks we are going to split the tensor into
        num_chunks = tf.math.ceil(tf.divide(inputs.shape[1], stride, name=self.name), name=self.name)
        num_chunks = tf.cast(num_chunks, tf.int32, name=self.name)  # cast necessary for tf operations
        # pad to get a multiple of stride + block_size as length (i.e. make sure that the last chunk has the correct size for concat)
        t = tf.pad(inputs, [[0, 0], [0, tf.subtract(tf.add(tf.multiply(num_chunks, stride, name=self.name), block_size, name=self.name), inputs.shape[1],
                                                    name=self.name)], [0, 0]], name=self.name)
        for i in range(num_chunks.numpy()):
            # extract stride
            t_stride = t[:, i * stride:i * stride + block_size, :]
            # add time dimension
            t_stride = tf.expand_dims(t_stride, 1, name=self.name)
            # concat
            if i == 0:
                t_new = t_stride
            else:
                t_new = tf.concat([t_new, t_stride], axis=1, name=self.name)
        t = t_new
        self.out_shape = t.shape
        return t

    def compute_output_shape(self, input_shape):
        return self.out_shape

    def get_config(self):
        config = {
            'block_size': self.block_size,
            'stride': self.stride
        }
        base_config = super(SplitLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class StandardizeLayer(Layer):

    # input shape is (batch, num_frames_per_time_step, num_points_per_frame, num_feats)
    # skip_standardize is a list of features to exclude from standardization (e.g. frame number)
    # standardize the points by considering each frame independently

    def __init__(self, eps=1e-6, skip_standardize=[0,1], name="StandardizeLayer", **kwargs):
        self.eps = eps
        self.skip_standardize = skip_standardize
        super(StandardizeLayer, self).__init__(name=name, **kwargs)
        self.trainable = False

    def call(self, inputs):
        feat_axis = 3
        for i in range(inputs.shape[feat_axis]):
            if i in self.skip_standardize:
                p = inputs[:,:,:,i]
                p = tf.expand_dims(p, -1)
                if i == 0:
                    t = p
                else:
                    t = tf.concat([t, p], axis=feat_axis)
            else:
                p = (inputs[:,:,:,i] - tf.math.reduce_mean(inputs[:,:,:,i], axis=feat_axis-1, keepdims=True)) / (tf.math.reduce_std(inputs[:,:,:,i], axis=feat_axis-1, keepdims=True) + self.eps)
                p = tf.expand_dims(p, -1)
                if i == 0:
                    t = p
                else:
                    t = tf.concat([t, p], axis=feat_axis)
        return t

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        return input_shape

    def get_config(self):
        config = {
            'eps': self.eps,
            'skip_standardize': self.skip_standardize
        }
        base_config = super(StandardizeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ExtractFeaturesLayer(Layer):

    # keep only a subset of features (a simple tf.gather)

    def __init__(self, keep_features=[2,3,4], axis=-1, name="ExtractFeaturesLayer", **kwargs):
        self.keep_features = keep_features
        self.axis = axis
        super(ExtractFeaturesLayer, self).__init__(name=name, **kwargs)
        self.trainable = False

    def call(self, inputs):
        return tf.gather(inputs, self.keep_features, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        input_shape[-1]=len(self.keep_features)
        return input_shape

    def get_config(self):
        config = {
            'keep_features': self.keep_features,
            'axis': self.axis
        }
        base_config = super(ExtractFeaturesLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    