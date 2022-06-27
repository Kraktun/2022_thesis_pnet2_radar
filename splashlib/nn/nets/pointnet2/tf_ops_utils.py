import tensorflow as tf
import numpy as np


# NOTE: all input data of these functions must be in the format NHWC
# refer to https://github.com/yanx27/Pointnet_Pointnet2_pytorch for the original conversion

@tf.function
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    B = tf.shape(src)[0]
    _, M, _ = dst.shape
    dist = -2 * tf.linalg.matmul(src, tf.transpose(dst, perm=[0, 2, 1]))
    dist += tf.reshape(tf.math.reduce_sum(src ** 2, axis=-1), shape=[B, N, 1])
    dist += tf.reshape(tf.math.reduce_sum(dst ** 2, axis=-1), shape=[B, 1, M])
    return dist

@tf.function
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    new_points = tf.gather(points, idx, axis = 1, batch_dims=1)
    return new_points

@tf.function
def farthest_point_sample(xyz, npoint, replace_inf=False):
    # NOTE: this is an approximate solution
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    B = tf.shape(xyz)[0]
    distance = None
    if replace_inf:
        # assume that all finite points are at the beginning
        t_size = tf.where(tf.math.is_finite(xyz[...,0]), [1], [0])
        t_size = tf.math.reduce_sum(t_size, axis=-1)
        t_size = tf.math.reduce_min(t_size)
        farthest = tf.random.uniform(minval=0, maxval=t_size, shape=(B,), dtype=tf.int32)
    else:
        farthest = tf.random.uniform(minval=0, maxval=N, shape=(B,), dtype=tf.int32)
    
    for i in range(npoint):
        if i == 0:
            centroids = farthest
            centroids = tf.expand_dims(centroids, -1)
        else:
            centroids = tf.concat([centroids, tf.expand_dims(farthest, -1)], axis=-1)
        centroid = tf.gather(xyz, farthest, axis = 1, batch_dims=1)
        centroid = tf.expand_dims(centroid, 1)
        dist = tf.math.reduce_sum(tf.math.subtract(xyz, centroid) ** 2, axis=-1)
        if replace_inf:
            # set to negative the distance when we consider inf points, to avoid choosing inf points
            dist = tf.where(tf.math.is_finite(dist), dist, [-1.0]) 
        if distance == None:
            distance = dist
        else:
            distance = tf.math.minimum(dist, distance)
        farthest = tf.math.argmax(distance, axis=-1, output_type=tf.int32)
    return centroids

@tf.function
def filter_unique(xyz):
    unique_sqrdists = tf.cast(square_distance(xyz, xyz), tf.float32)[...,0]
    unique_sqrdists = tf.expand_dims(unique_sqrdists, -1)
    unique_sqrdists = tf.tile(unique_sqrdists, [1, xyz.shape[-1]])
    unique_mask = unique_sqrdists <= (1e-9)
    tensor_out = tf.where(unique_mask, xyz, [np.inf])
    return tensor_out

@tf.function
def query_ball_point(radius, nsample, xyz, new_xyz, unique=False):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    
    unique here does not actually remove all duplicate points, but removes only the duplicates of the centroids.
    To remove all duplicates you should use the filter_unique function that is however pretty inefficient.
    """
    B, N, C = xyz.shape
    B = tf.shape(xyz)[0]
    _, S, _ = new_xyz.shape
    group_idx = tf.tile(tf.reshape(tf.range(N, dtype=tf.int32), shape=[1, 1, N]), [B, S, 1])
    # xyz are the points, new_xyz are the centroids
    # if unique:
    #     # keeps the same shape, but maps the duplicate points to inf
    #     xyz = tf.map_fn(filter_unique, xyz)
    # compute the distance between each pair of points
    sqrdists = tf.cast(square_distance(new_xyz, xyz), tf.float32)
    # find where it's < radius
    mask = sqrdists <= (radius ** 2)
    # group_idx contains the index of the points that are at distance < radius, otherwise N (which is outside the index range)
    group_idx = tf.where(mask, group_idx, N) # where with 3 elements: where it's true take the value in group_idx, N otherwise
    if unique:
        # mask to remove duplicate points (i.e. same distance)
        mask2 = sqrdists > 0
        group_idx_red = tf.where(mask2, group_idx, N)
    # take only a subset of points, if limited by nsample
    group_idx = tf.sort(group_idx, axis=-1)[:, :, :nsample]
    # repeat the indexes, to be sure you have enough to reach nsample
    group_first = tf.tile(tf.reshape(group_idx[:, :, 0], shape=[B, S, 1]), [1, 1, nsample])
    if unique:
        group_idx_red = tf.sort(group_idx_red, axis=-1)[:, :, :nsample]
        group_idx = group_idx_red
    # same mask as before, but opposite values, also limited to nsample
    mask = group_idx == N
    # now, take nsample from the points with distance < radius, and if you have not enough take again from those that have distance < radius
    group_idx = tf.where(mask, group_first, group_idx)
    return group_idx

@tf.function
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, unique=False, replace_inf=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
        replace_inf: if true, set to 0 the features that correspond to invalid (inf) points and avoid inf points in fps (if possible)
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    B = tf.shape(xyz)[0]
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint, replace_inf) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz, unique)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - tf.reshape(new_xyz, shape=[B, S, 1, C])

    if points is not None:
        grouped_points = index_points(points, idx)
        if len(tf.shape(grouped_points)) < len(tf.shape(grouped_xyz_norm)):
            grouped_points = tf.expand_dims(grouped_points, axis=-1)
        if replace_inf:
            # set to 0 where it's NaN, this should not be necessary
            grouped_points = tf.where(tf.math.is_finite(grouped_points), grouped_points, [0.0])
        new_points = tf.concat([grouped_xyz_norm, grouped_points], axis=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

@tf.function
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    B, N, C = xyz.shape
    B = tf.shape(xyz)[0]
    new_xyz = tf.zeros([B, 1, C])
    grouped_xyz = tf.reshape(xyz, shape=[B, 1, N, C])
    if points is not None:
        new_points = tf.concat([grouped_xyz, tf.reshape(points, shape=[B, 1, N, -1])], axis=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
    