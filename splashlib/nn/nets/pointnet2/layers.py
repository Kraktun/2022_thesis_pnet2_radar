import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv1D
from tensorflow.python.framework import tensor_shape

from .tf_ops_utils import *

# NOTE: This is hardcoded to use data_format='channel_last'

class PointNetSetAbstraction(Layer):

    def __init__(self, npoint, radius, nsample, filters, group_all, replace_inf, bn=False, unique=True, verbose=False, name="PointNetSetAbstraction", **kwargs):
        super(PointNetSetAbstraction, self).__init__(name=name, **kwargs)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.mlps = []
        for filter in filters:
            self.mlps.append(MLP(filters=filter, kernel_size=1, strides=1, activation='relu', bn=self.bn, data_format="channels_last"))
        self.group_all = group_all
        self.replace_inf = replace_inf
        self.verbose = verbose
        self.unique = unique

    def call(self, xyz, points):
        """
        Input:
            xyz: input points position data, NHWC
            points: input points data, NHWC
        Return:
            new_xyz: sampled points position data, NHWC
            new_points_concat: sample points feature data, NHWC
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, unique=self.unique, replace_inf=self.replace_inf)
        
        new_points = tf.transpose(new_points, perm=[0,2,1,3]) # [B, nsample, npoint, C+D]
        for conv in self.mlps:
            new_points =  conv(new_points)

        if self.verbose:
            tf.print(new_points)
        new_points = tf.math.reduce_max(new_points, axis=1) # [B, npoint, C+D]
        return new_xyz, new_points
    
    def get_config(self):
        config = {
            'npoint': self.npoint, 
            'radius': self.radius, 
            'nsample': self.nsample, 
            'mlps': self.mlps,
            'group_all': self.group_all
        }
        base_config = super(PointNetSetAbstraction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PointNetSetAbstractionMsg(Layer):

    # Multi-scale-grouping, see paper on pointnet++ sec. 3.3

    # NOT TESTED

    def __init__(self, npoint, radius_list, nsample_list, in_channel, filters_list, name="PointNetSetAbstractionMsg", **kwargs):
        super(PointNetSetAbstractionMsg, self).__init__(name=name, **kwargs)
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlps_list = []
        for i in range(len(filters_list)):
            mlps = []
            for filter in filters_list[i]:
                mlps.append(MLP(filters=filter, kernel_size=1, strides=1, activation='relu', bn=True, data_format="channels_last"))
            self.mlps_list.append(mlps)

    def call(self, xyz, points):
        """
        Input:
            xyz: input points position data, NHWC
            points: input points data, NHWC
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        
        B, N, C = xyz.shape
        B = tf.shape(xyz)[0]
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= tf.reshape(new_xyz, shape=[B, S, 1, C])
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = tf.reshape(grouped_points, shape=[0, 2, 1, 3])  # [B, K, S, D]
            for j in range(len(self.mlps_list[i])):
                grouped_points =  self.mlps_list[i][j](grouped_points)
            new_points = tf.math.reduce_max(grouped_points, axis=1)[0]  # [B, S, D']
            new_points_list.append(new_points)

        new_points_concat = tf.concat(new_points_list, axis=1)
        return new_xyz, new_points_concat
    
    def get_config(self):
        config = {
            'npoint': self.npoint,
            'radius_list': self.radius_list,
            'nsample_list': self.nsample_list,
            'mlps_list': self.mlps_list,
        }
        base_config = super(PointNetSetAbstractionMsg, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PointNetFeaturePropagation(Layer):
    def __init__(self, mlp, bn=False, name="PointNetFeaturePropagation", **kwargs):
        super(PointNetFeaturePropagation, self).__init__(name=name, **kwargs)
        self.bn = bn
        self.mlp_convs = []
        self.mlp_bns = []
        for out_channel in mlp:
            self.mlp_convs.append(Conv1D(filters=out_channel, kernel_size=1, activation="relu"))
            self.mlp_bns.append(BatchNormalization())

    def call(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N] NHWC
            xyz2: sampled input points position data, [B, C, S] NHWC
            points1: input points data, [B, D, N] NHWC
            points2: input points data, [B, D, S] NHWC
        Return:
            new_points: upsampled points data, [B, D', N] BND'
        """
        # 2 is the smaller set of points (i.e. the centroids of 1),
        # here we take the feats of 1 (the bigger set) and append the feats of 2 weighted by the 
        # distances of the centroids from the points in 1
        B, N, C = xyz1.shape
        B = tf.shape(xyz1)[0]
        S = xyz2.shape[1]

        if S == 1:
            interpolated_points = tf.repeat(points2, repeats=[1, N, 1])
        else:
            dists = square_distance(xyz1, xyz2)
            idx = tf.argsort(dists, axis=-1)
            dists = tf.gather(dists, idx, batch_dims=-1) # batch_dims=-1 necessary with idx from argsort
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = tf.reduce_sum(dist_recip, axis=2, keepdims=True)
            weight = dist_recip / norm
            interpolated_points = tf.reduce_sum(index_points(points2, idx) * tf.reshape(weight, shape=[B, N, 3, 1]), axis=2)

        if points1 is not None:
            new_points = tf.concat([points1, interpolated_points], axis=-1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            new_points = conv(new_points)
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = bn(new_points)
        return new_points
    
    def get_config(self):
        config = {
            'mlp_convs': self.mlp_convs,
            'mlp_bns': self.mlp_bns
        }
        base_config = super(PointNetFeaturePropagation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MLP(Layer):
    def __init__(self, filters, kernel_size=[1, 1], strides=[1, 1], padding='valid', data_format='channels_last', 
                    activation='relu', bn=False, name="MLP", **kwargs):
        super(MLP, self).__init__(name=name, **kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.activation = activation
        self.bn = bn

        if self.data_format == 'channels_last':
            self.axis = -1
        elif self.data_format == 'channels_first':
            self.axis = 1

    def build(self, input_shape):
        self.conv = Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, 
                        data_format=self.data_format, activation=self.activation)
        if self.bn:
            self.batch_norm = BatchNormalization(axis=self.axis)
        super(MLP, self).build(input_shape)

    def call(self, inputs):
        output = self.conv(inputs)
        if self.bn:
            output = self.batch_norm(output)

        return output

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': self.activation
        }
        base_config = super(MLP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        # same as conv2d, bn does not change the shape
        return Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                             data_format=self.data_format, activation=self.activation).compute_output_shape(input_shape)