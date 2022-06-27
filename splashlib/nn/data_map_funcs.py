import tensorflow as tf
import numpy as np

from splashlib.nn.k_utils import *


def apply_dataset_shift(dataset, dataset_shift_append=True, dataset_shift_interval=[-1,1], dataset_shift_prob=0.3, **dataset_vars):
    """
    Apply a shift to both point clouds and markers (the shift is the same for each sequence, 
    but different among sequences).

    dataset: input dataset
    dataset_shift_append: set to True if you want the shifted dataset to be appended to the original one
        rather than replace it
    dataset_shift_interval: interval for the shift to be added
    dataset_shift_prob: probability of the shift to be applied to each sequence (not per point, but per sequence)
    """
    if dataset is None:
        return None

    def apply_shift(x, y):
        if np.random.random() <= dataset_shift_prob:
            shifts = tf.random.uniform(shape=(3,), minval=dataset_shift_interval[0], maxval=dataset_shift_interval[1])
            # can't use tile as I can't manipulate the shape from tf.shape
            # but you can use it if you infer the shape first by extracting a value
            def extend_to_size(t, t_shifts):
                if tf.shape(t)[-1] > 3: # add a zero column at the end
                    t_shifts = tf.concat([t_shifts, [0]], axis=-1)
                t_shifts = tf.expand_dims(t_shifts,0)
                t_shifts = tf.expand_dims(t_shifts,0)
                t_shifts = tf.repeat(t_shifts, tf.shape(t)[0], axis=0)
                t_shifts = tf.repeat(t_shifts, tf.shape(t)[1], axis=1)
                return t_shifts
            
            x_shifts = extend_to_size(x, shifts)
            y_shifts = extend_to_size(y, shifts)
            return tf.math.add(x, x_shifts), tf.math.add(y, y_shifts)
        else:
            return x,y
    
    shifted_dataset = dataset.map(apply_shift)
    if dataset_shift_append:
        shifted_dataset = dataset.concatenate(shifted_dataset)
    
    return shifted_dataset

def apply_dataset_rotation(dataset, dataset_rotation_append=True, dataset_rotation_intervals=[-1,1]*3, dataset_rotation_prob=0.3, **dataset_vars):
    """
    Apply a rotation to both point clouds and markers (the rotation is the same for each sequence, 
    but different among sequences).

    dataset: input dataset
    dataset_rotation_append: set to True if you want the rotated dataset to be appended to the original one
        rather than replace it
    dataset_rotation_intervals: intervals for the rotation to be added. The value is multiplied by 2*pi, so an interval
        [-1,1] results in a rotation of [-2pi, 2pi]
    dataset_rotation_prob: probability of the rotation to be applied to each sequence (not per point, but per sequence)
    """
    if dataset is None:
        return None

    def apply_rotation(x, y):
        if np.random.random() <= dataset_rotation_prob:
            angle_x = np.random.random()*(dataset_rotation_intervals[0][1]-dataset_rotation_intervals[0][0]) + dataset_rotation_intervals[0][0]
            angle_x = angle_x*np.pi*2
            angle_y = np.random.random()*(dataset_rotation_intervals[1][1]-dataset_rotation_intervals[1][0]) + dataset_rotation_intervals[1][0]
            angle_y = angle_y*np.pi*2
            angle_z = np.random.random()*(dataset_rotation_intervals[2][1]-dataset_rotation_intervals[2][0]) + dataset_rotation_intervals[2][0]
            angle_z = angle_z*np.pi*2

            sin_x = np.sin(angle_x)
            cos_x = np.cos(angle_x)
            sin_y = np.sin(angle_y)
            cos_y = np.cos(angle_y)
            sin_z = np.sin(angle_z)
            cos_z = np.cos(angle_z)

            def process_tensor(t):
                t_x = t[...,0]
                t_y = t[...,1]
                t_z = t[...,2]
                t_other = t[...,3:]
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
                if tf.shape(t)[-1] > 3:
                    t_all = tf.concat([t_all, t_other], axis=-1)
                return t_all
            
            x_rots = process_tensor(x)
            y_rots = process_tensor(y)
            return x_rots, y_rots
        else:
            return x,y
    
    rotated_dataset = dataset.map(apply_rotation)
    if dataset_rotation_append:
        rotated_dataset = dataset.concatenate(rotated_dataset)
    
    return rotated_dataset


def apply_center_sequence(dataset, dataset_center_seq_mode="current", dataset_center_seq_dims=None, **dataset_vars):
    """
    Shift a sequence (both cloud points and markers) to the center 0, where the center of the sequence is computed 
    according to dataset_center_seq_mode on the dimensions specified by dataset_center_seq_dims
    The center is computed on the point clouds, not on the markers and is the same for both clouds and markers
    

    dataset: input dataset
    dataset_center_seq_mode: string among ['all', 'first', 'middle', 'last', 'current', 'previous'] to indicate 
        which frames to use to compute the center of the clouds.
    dataset_center_seq_dims: list of the dimensions to use (of the last axis). Usually [0,1,2] for x,y,z
    """
    # Pretty much no comments on the code as it was one of the last things I wrote.
    # Trust me, it works.
    if dataset is None:
        return None
    
    assert dataset_center_seq_mode in ['all', 'first', 'middle', 'last', 'current', 'previous']

    center_mode = dataset_center_seq_mode
    use_dims = dataset_center_seq_dims

    # manually extract shape or the code breaks
    for x,y in dataset.take(1):
        x_shape = x.shape
        y_shape = y.shape

    def center_seq(x, y):
        feat_centers = []
        if use_dims is None:
            dims_arr = range(x_shape[-1])
        else:
            dims_arr = use_dims
        if center_mode == "all":
            for dim in dims_arr:
                centroid = finite_reduce_mean(x[...,dim], axis=-1, keepdims=True) # reduce for all points in a frame
                centroid = tf.math.reduce_mean(centroid, axis=-2, keepdims=True) # reduce for all frames in a sequence
                feat_centers.append(centroid)
            centroids = tf.stack(feat_centers, axis=-1)
            x_centroids = tf.tile(centroids, [x_shape[-3],x_shape[-2],1])
            y_centroids = tf.tile(centroids, [y_shape[-3],y_shape[-2],1])
        elif center_mode == "first" or center_mode == "middle" or center_mode == "last":
            if center_mode == "first":
                take_frame = 0
            elif center_mode == "middle":
                take_frame = x_shape[-3]//2
            elif center_mode == "last":
                take_frame = -1
            else:
                raise ValueError("Invalid center mode in CenterSequence")
            for dim in dims_arr:
                centroid = finite_reduce_mean(x[...,dim], axis=-1, keepdims=True) # reduce for all points in a frame
                centroid = centroid[take_frame] # take chosen frame
                centroid = tf.expand_dims(centroid,-2) # add axis
                feat_centers.append(centroid)

            centroids = tf.stack(feat_centers, axis=-1)
            x_centroids = tf.tile(centroids, [x_shape[-3],x_shape[-2],1])
            y_centroids = tf.tile(centroids, [y_shape[-3],y_shape[-2],1])
        elif center_mode == "current" or center_mode == "previous":
            for dim in dims_arr:
                centroid = finite_reduce_mean(x[...,dim], axis=-1, keepdims=True) # reduce for all points in a frame: 30,1,1
                feat_centers.append(centroid)
            centroids = tf.stack(feat_centers, axis=-1) # 30, 1, len(dims_arr)
            if center_mode == "previous":
                centroids = tf.concat([tf.expand_dims(centroids[0],0), centroids[:-1]], axis=0)
            x_centroids = tf.tile(centroids, [1,x_shape[-2],1])
            if y_shape[-3] < x_shape[-3]:
                centroids = tf.expand_dims(centroids[-1], 0)
            y_centroids = tf.tile(centroids, [1,y_shape[-2],1])
            
        last_centroid_used = -1
        if use_dims is None:
            final_result = (x - x_centroids, y - y_centroids)
        else:
            last_centroid_used = -1
            for i in range(x_shape[-1]):
                if i in dims_arr:
                    last_centroid_used = last_centroid_used+1
                    x_column_to_add = x[...,i]-x_centroids[...,last_centroid_used]
                else:
                    x_column_to_add = x[...,i]
                if i == 0:
                    x_final_result = tf.expand_dims(x_column_to_add, -1)
                else:
                    x_column_to_add = tf.expand_dims(x_column_to_add, -1)
                    x_final_result = tf.concat([x_final_result, x_column_to_add], axis=-1)
            last_centroid_used = -1
            for i in range(y_shape[-1]):
                if i in dims_arr:
                    last_centroid_used = last_centroid_used+1
                    y_column_to_add = y[...,i]-y_centroids[...,last_centroid_used]
                else:
                    y_column_to_add = y[...,i]
                if i == 0:
                    y_final_result = tf.expand_dims(y_column_to_add, -1)
                else:
                    y_column_to_add = tf.expand_dims(y_column_to_add, -1)
                    y_final_result = tf.concat([y_final_result, y_column_to_add], axis=-1)
            final_result = (x_final_result, y_final_result)
        return final_result

    dataset = dataset.map(center_seq)
    return dataset

def apply_dataset_normalization(dataset, dataset_normalization_axis=[0,1,2], dataset_normalization_xyz=False, dataset_normalization_sequence=[False,False,False], **dataset_vars):
    """
    Apply a normalization to both point clouds and markers to a [0,1] interval.
    The normalization can be performed in different ways: per sequence, per frame, on a single axis, on multiple axis

    dataset: input dataset
    dataset_normalization_axis: on which axis to apply normalization. With axis I mean on which features, so a list of
        ints, for instance [0,1,2] means x,y,z; [2,3] means z and velocity etc.
    dataset_normalization_xyz: True to normalize the x,y,z axis together, i.e. build the interval by using 
        the min(x_min, y_min, z_min) and max(x_max, y_max, z_max)
    dataset_normalization_sequence: list of the same length of dataset_normalization_axis to normalize the axis where
        dataset_normalization_sequence is True on the whole sequence rather than on each frame
    """
    if dataset is None:
        return None

    if isinstance(dataset_normalization_sequence, bool):
        dataset_normalization_sequence = [dataset_normalization_sequence]*len(dataset_normalization_axis)

    # manually extract shape or the code breaks
    for x,y in dataset.take(1):
        x_shape = x.shape
        y_shape = y.shape
    
    def apply_normalization(x, y):
        # use normalization for x also for y
        x_ax_min = tf.math.reduce_min(x, axis=-2, keepdims=True)
        x_ax_max = tf.math.reduce_max(x, axis=-2, keepdims=True)
        # normalize: (x-min)/(max-min)
        if dataset_normalization_xyz:
            # use only xyz axis
            x_ax_min_xyz = tf.math.reduce_min(x_ax_min[...,:3], axis=-1, keepdims=True)
            x_ax_min_xyz = tf.repeat(x_ax_min_xyz, 3, axis=-1)
            x_ax_min = tf.concat([x_ax_min_xyz, x_ax_min[...,3:]], axis=-1)
            x_ax_max_xyz = tf.math.reduce_max(x_ax_max[...,:3], axis=-1, keepdims=True)
            x_ax_max_xyz = tf.repeat(x_ax_max_xyz, 3, axis=-1)
            x_ax_max = tf.concat([x_ax_max_xyz, x_ax_max[...,3:]], axis=-1)
        x_ax_min_t = []
        x_ax_max_t = []
        curr_norm = 0
        for ax in range(x_shape[-1]):
            if ax not in dataset_normalization_axis:
                x_ax_min_t.append(x_ax_min[...,ax])
                x_ax_max_t.append(x_ax_max[...,ax])
                continue
            if dataset_normalization_sequence[curr_norm]:
                if ax == 1: # for y ignore the frames with (0,0) 
                    t_x_ax_min = x_ax_min[...,ax]
                    t_x_ax_min = tf.where(t_x_ax_min>0.0, t_x_ax_min, 1e9)
                else:
                    t_x_ax_min = x_ax_min[...,ax]
                tp = tf.math.reduce_min(t_x_ax_min, axis=0, keepdims=True)
                tp = tf.tile(tp, [x_shape[0],1])
                x_ax_min_t.append(tp)
                tp = tf.math.reduce_max(x_ax_max[...,ax], axis=0, keepdims=True)
                tp = tf.tile(tp, [x_shape[0],1])
                x_ax_max_t.append(tp)
            else:
                x_ax_min_t.append(x_ax_min[...,ax])
                x_ax_max_t.append(x_ax_max[...,ax])
            curr_norm = curr_norm+1
        x_ax_min = tf.stack(x_ax_min_t, axis=-1)
        x_ax_max = tf.stack(x_ax_max_t, axis=-1)
        x_ax_interval = x_ax_max - x_ax_min
        # replace 0 values with 1 to avoid NaN in the division
        x_ax_interval = tf.where(x_ax_interval==0.0, 1.0, x_ax_interval)
        y_ax_min = x_ax_min
        y_ax_max = x_ax_max
        if y_shape[0] < x_shape[0]:
            y_ax_min = y_ax_min[-y_shape[0]]
            y_ax_max = y_ax_max[-y_shape[0]]
            if len(y_ax_min.shape) < len(y_shape):
                y_ax_min = tf.expand_dims(y_ax_min, 0)
                y_ax_max = tf.expand_dims(y_ax_max, 0)
        y_ax_interval = y_ax_max - y_ax_min
        # replace 0 values with 1 to avoid NaN in the division
        y_ax_interval = tf.where(y_ax_interval==0.0, 1.0, y_ax_interval)
        x_rev = []
        y_rev = []
        for ax in range(x_shape[-1]):
            x_new = x[...,ax]
            if ax in dataset_normalization_axis:
                t_x_min = x_ax_min[...,ax]
                t_x_int = x_ax_interval[...,ax]
                x_new = (x_new-t_x_min)/t_x_int
            x_rev.append(x_new)
        for ax in range(y_shape[-1]):
            y_new = y[...,ax]
            if ax in dataset_normalization_axis and ax < 3: # only normalize xyz, not the column to enable/disable markers
                t_y_min = y_ax_min[...,ax]
                t_y_int = y_ax_interval[...,ax]
                y_new = (y_new-t_y_min)/t_y_int
            y_rev.append(y_new)
        
        x_rev = tf.stack(x_rev, axis=-1)
        y_rev = tf.stack(y_rev, axis=-1)
        return x_rev, y_rev
    
    normalized_dataset = dataset.map(apply_normalization)
    
    return normalized_dataset
