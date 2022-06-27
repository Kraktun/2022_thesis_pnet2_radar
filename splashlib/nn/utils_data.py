import re
import pandas as pd
import os
import tensorflow as tf
from pathlib import Path
import numpy as np


def load_csv_file(file_path, select_from=0, select_to=None, separator=','):
    """
    Read a csv file and return a tensor with shape (num_rows, num_cols)
    note that it's a single tensor, not a list of tensors
    file_path can be either a string or a string tensor
    set select_from and select_to to keep only a subset of lines, e.g. select_to=-1 
    to remove an empty line at the end
    """
    raw = tf.io.read_file(file_path)
    # to a list of rows
    fl = tf.strings.split(raw, '\n')
    # for compatibility with datasets generated in windows, remove carriage return
    ff = tf.map_fn(fn=lambda t: tf.strings.regex_replace(t, '\r*', ''), elems=fl) 
    # select only required rows
    ff = ff[select_from:select_to]
    # split numbers
    ff = tf.map_fn(fn=lambda t: tf.strings.split(t, separator), elems=ff)
    # convert strings to floats
    ff = tf.strings.to_number(ff, tf.float32)
    # stack numbers (i.e. map to rows)
    ff = tf.map_fn(tf.stack, elems=ff)
    # concat to create the row axis
    ff = tf.concat(ff, axis=0)
    return ff


def build_dataset(dataset_path):
    """
    load the csv files from a given path and maps them to a tensorflow dataset
    to collect in a split dataset also the csv must be split (i.e. one csv for each frame)
    return a list of datasets where each dataset contains the frames in a single sequence
    """
    dirs = os.listdir(dataset_path)
    datasets = []
    for d in dirs:
        # each dir is a track of multiple frames
        filenames = tf.io.gfile.glob(os.path.join(dataset_path, d, "*.csv"))
        filenames = sorted(filenames)
        # map list of filenames to a tf dataset
        ff = tf.data.Dataset.from_tensor_slices(filenames)
        # load each csv to a tensor with shape (num_points_per_frame, num_features)
        ff = ff.map(lambda x: load_csv_file(x, select_to=-1))
        datasets.append(ff)
    return datasets

def split_dataset(dataset, block_length, stride):
    """
    input is a tf dataset with tensors with shape (num_points_per_frame, num_features)
    this method groups the tensors in windows of size block_length with a stride of size stride
    which is: if the index of the first tensor of window i is m, the first tensor of the following window (i+1) will be m+stride
    blocks will have overlapping tensors is stride < block_length
    if dataset length is not a multiple of stride, the remaining tensors (that don't complete a window) will be discarded
    """
    # discard the remaining tensors that don't fit
    num_blocks = np.floor(len(dataset) / stride).astype(np.int32)
    ds_m = dataset.take(num_blocks*stride)
    # add row dimension
    ds_m = ds_m.map(lambda x: tf.expand_dims(x, axis=0))
    
    # concat all the tensors in a single tensor with shape (total_number_of_blocks, num_points_per_frame, num_features)
    if tf.__version__ < '2.6.0':
        in_tensor = tf.data.experimental.get_single_element(ds_m.take(1))
    else:
        in_tensor = ds_m.take(1).get_single_element()
    ds_m = ds_m.skip(1).reduce(in_tensor, lambda agg, new: tf.concat([agg, new], axis = 0))
    
    # divide the tensors in windows of size block_length
    for i in range(num_blocks):
        # extract slice
        t_stride = ds_m[i * stride:i * stride + block_length, :, :]
        # concat
        if i == 0:
            t_new = t_stride
        else:
            t_new = tf.concat([t_new, t_stride], axis=0)
    return tf.data.Dataset.from_tensor_slices(ds_m)


def split_dataset_2(dataset, block_length, shift):
    """
    input is a tf dataset with tensors with shape (num_points_per_frame, num_features)
    this method groups the tensors in windows of size block_length with a stride of size shift
    which is: if the index of the first tensor of window i is m, the first tensor of the following window (i+1) will be m+stride
    blocks will have overlapping tensors is stride < block_length
    if dataset length is not a multiple of stride, the remaining tensors (that don't complete a window) will be discarded
    """
    # obtain windows, note that the windows are VariantDatasets and need to be converted to tensors
    ds_m = dataset.window(size=block_length, shift=shift, drop_remainder=True)
    
    # convert each dataset to a single tensor with shape (block_length, num_points_per_frame, num_features)
    def collect_dataset(x):
        x = x.map(lambda p: tf.expand_dims(p, 0))
        if tf.__version__ < '2.6.0':
            in_tensor = tf.data.experimental.get_single_element(x.take(1))
        else:
            in_tensor = x.take(1).get_single_element()
        x = x.skip(1).reduce(in_tensor, lambda agg, new: tf.concat([agg, new], axis = 0))
        return x
    
    ds_m = ds_m.map(lambda x: collect_dataset(x))
            
    return ds_m


def dataset_list_to_split(datasets, block_length, stride, version="2"):
    """
    take as input a list of tf datasets and
    split each dataset according to the provided block length and stride
    then concatenate in a single dataset where each tensor has shape (block_length, num_points_per_frame, num_features)
    version determines the grouping strategy (results are the same, complexity may vary)
    """
    if version == "1":
        ds = [split_dataset(t, block_length, stride) for t in datasets]
    elif version == "2":
        ds = [split_dataset_2(t, block_length, stride) for t in datasets]
    else:
        raise ValueError('Invalid version.')
    # extract the first element for concatenation, then loop across all the others
    # these are datasets, not tensors
    t = ds[0]
    ds = ds[1:]
    for c in ds:
        t = t.concatenate(c)
    return t


def save_dataset(dataset, time_steps, time_steps_stride, dataset_folder, as_records=False, subfolder="train"):
    """
    Save a dataset either as tf records or with tf.data.experimental.save
    Time steps and strides are necessary only to create the correct folder. The dataset is saved as is.
    Final folder will be $dataset_folder/records/$time_steps_$time_steps_stride is saved as records
    $dataset_folder/save/$time_steps_$time_steps_stride/$subfolder/ otherwise
    """
    # dataset must be the result of dataset_list_to_split
    if as_records:
        folder_path = os.path.join(dataset_folder, "records", f"{time_steps}_{time_steps_stride}")
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(folder_path, f"{subfolder}.tfrecord")
        mm_ds = dataset.map(tf.io.serialize_tensor)
        # convert dataset to numpy list
        mm_ds = [x.numpy() for x in mm_ds]
        with tf.io.TFRecordWriter(file_path, options=tf.io.TFRecordOptions(compression_type='GZIP')) as w:
            for m in mm_ds:
                w.write(m)
    else:
        folder_path = os.path.join(dataset_folder, "save", f"{time_steps}_{time_steps_stride}", subfolder)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        _ = tf.data.experimental.save(dataset, folder_path, compression='GZIP')


def load_dataset(time_steps, time_steps_stride, dataset_folder, as_records=False, subfolder="train"):
    """
    Load a dataset from a given folder. Check save_dataset for the naming convention.
    """
    if as_records:
        folder_path = os.path.join(dataset_folder, "records", f"{time_steps}_{time_steps_stride}")
        file_path = os.path.join(folder_path, f"{subfolder}.tfrecord")
        if not Path(file_path).exists():
            return None
        return tf.data.TFRecordDataset(file_path, compression_type='GZIP').map(lambda x: tf.io.parse_tensor(x, tf.float32))
    else:
        folder_path = os.path.join(dataset_folder, "save", f"{time_steps}_{time_steps_stride}", subfolder)
        if not Path(folder_path).exists() or tf.__version__ < '2.5.0': # current saving and loading is supported only in tf >= 2.5
            return None
        return tf.data.experimental.load(folder_path, compression='GZIP')


def setup_dataset(dataset_dir, 
        val_dir='',
        reduce_train_set=0, 
        skip_train_set=0, 
        reduce_val_set=0, 
        skip_val_set=0, 
        time_steps=None, 
        time_steps_stride=None,
        val_ds_from_train=False,
        val_ds_percent=0.2,
        **marker_vars): # dump marker specific parameters
    """
    All in one function to load the dataset, this is the function to call to load/generate 
    the correct dataset according to the variables provided with cli.
    If you want to save a dataset, do not use the skip or reduce parameters.

    dataset_dir: main directory of the train dataset
    val_dir: main directory of the validation dataset
    reduce_train_set: if int > 0 use only a subset of elements from the training set 
    skip_train_set: if int > 0 skip this number of elements from the training set 
    reduce_val_set: if int > 0 use only a subset of elements from the validation set 
    skip_val_set: if int > 0 skip this number of elements from the validation set 
    time_steps: time steps for this dataset
    time_steps_stride: strides for this dataset
    val_ds_from_train: true to split train ds in train_ds, val_ds so that len(val_ds)=len(original_train_ds)*val_ds_percent
        Used only if val_dir is None, empty, or invalid
    val_ds_percent: percentage of samples from the training set to use for the validation set
    marker_vars: all the other variables that are not needed for this function
    """
    # check if an exported dataset is already available
    train_ds = load_dataset(time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=dataset_dir, as_records=False, subfolder="train")
    if train_ds is None:
        # check dataset saved as records
        train_ds = load_dataset(time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=dataset_dir, as_records=True, subfolder="train")
    if val_dir: # val_dir is provided (i.e. not empty)
        # note: subfolder remains "train"
        val_ds = load_dataset(time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=val_dir, as_records=False, subfolder="train")
        if val_ds is None:
            val_ds = load_dataset(time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=val_dir, as_records=True, subfolder="train")
    else:
        val_ds = None
    
    if val_dir:
        train_path = os.path.join(dataset_dir, 'train/')
        val_path = os.path.join(val_dir, 'train/')
    else:
        train_path = os.path.join(dataset_dir, 'train/')
        val_path = os.path.join(dataset_dir, 'val/')
    
    if train_ds is None:
        # build dataset from scratch
        train_ds = build_dataset(train_path) 
        if reduce_train_set > 0:
            # TAKE ONLY A SUBSET (note that this is an initial reduction, 
            # to avoid first part of computation, includes also the case where we want val_ds from train_ds)
            train_ds = train_ds[:reduce_train_set+skip_train_set+reduce_val_set+skip_val_set] 
        train_ds = dataset_list_to_split(train_ds, time_steps, time_steps_stride)
        # save in both formats
        if tf.__version__ >= '2.5.0':
            # current saving and loading is supported only in tf >= 2.5
            save_dataset(dataset=train_ds, time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=dataset_dir, as_records=False, subfolder="train")
        save_dataset(dataset=train_ds, time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=dataset_dir, as_records=True, subfolder="train")
    
    if val_ds is None and val_dir:
        # build val ds as a normal train ds 
        val_ds = build_dataset(val_path)
        if reduce_val_set > 0:
            val_ds = val_ds[:reduce_val_set+skip_val_set]
        val_ds = dataset_list_to_split(val_ds, time_steps, time_steps_stride)
        if tf.__version__ >= '2.5.0':
            save_dataset(dataset=val_ds, time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=val_dir, as_records=False, subfolder="train")
        save_dataset(dataset=val_ds, time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=val_dir, as_records=True, subfolder="train")
    
    # generate a val dataset from the train set if no specific validation dataset is available
    if val_ds is None and val_ds_from_train:
        train_ds_len = len(train_ds)
        val_ds_len = int(train_ds_len*val_ds_percent)
        train_ds_len = train_ds_len - val_ds_len
        val_ds = train_ds.skip(train_ds_len).take(val_ds_len)
        train_ds = train_ds.take(train_ds_len)

    # reduce datasets
    if reduce_train_set > 0:
        if skip_train_set + reduce_train_set > len(train_ds):
            raise ValueError("Requested more sample from the train set than available!")
        train_ds = train_ds.skip(skip_train_set).take(reduce_train_set)

    if reduce_val_set > 0 and val_ds is not None:
        if skip_val_set + reduce_val_set > len(val_ds):
            raise ValueError("Requested more sample from the val set than available!")
        val_ds = val_ds.skip(skip_val_set).take(reduce_val_set)

    return train_ds, val_ds


def setup_markers(dataset_dir, 
        val_dir='',
        reduce_train_set=0, 
        skip_train_set=0, 
        reduce_val_set=0, 
        skip_val_set=0, 
        time_steps=None, 
        time_steps_stride=None,
        val_ds_from_train=False, 
        val_ds_percent=0.2,
        marker_last_frame=False,
        **dataset_vars): # dump all other parameters
    """
    The code is a 1:1 copy of setup_dataset except for the name of the subfolder and the call to enable_for_ground_truth
    This function also includes marker_last_frame that if set to true reduces the markers dataset to the 
    last frame of each sequence.
    """
    train_ds = load_dataset(time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=dataset_dir, as_records=False, subfolder="markers")
    if train_ds is None:
        train_ds = load_dataset(time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=dataset_dir, as_records=True, subfolder="markers")
    train_path = os.path.join(dataset_dir, 'markers/')
    if train_ds is None:
        train_ds = build_dataset(train_path) 
        if reduce_train_set > 0:
            train_ds = train_ds[:reduce_train_set+skip_train_set+reduce_val_set+skip_val_set] 
        train_ds = dataset_list_to_split(train_ds, time_steps, time_steps_stride)
        if tf.__version__ >= '2.5.0':
            save_dataset(dataset=train_ds, time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=dataset_dir, as_records=False, subfolder="markers")
        save_dataset(dataset=train_ds, time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=dataset_dir, as_records=True, subfolder="markers")
    if val_dir: # i.e. not empty
        val_ds = load_dataset(time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=val_dir, as_records=False, subfolder="markers")
        if val_ds is None:
            val_ds = load_dataset(time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=val_dir, as_records=True, subfolder="markers")
        val_path = os.path.join(val_dir, 'markers/')
        if val_ds is None:
            val_ds = build_dataset(val_path) 
            if reduce_val_set > 0:
                val_ds = val_ds[:reduce_val_set+skip_val_set]
            val_ds = dataset_list_to_split(val_ds, time_steps, time_steps_stride)
            if tf.__version__ >= '2.5.0':
                save_dataset(dataset=val_ds, time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=val_dir, as_records=False, subfolder="markers")
            save_dataset(dataset=val_ds, time_steps=time_steps, time_steps_stride=time_steps_stride, dataset_folder=val_dir, as_records=True, subfolder="markers")
    elif val_ds_from_train:
        train_ds_len = len(train_ds)
        val_ds_len = int(train_ds_len*val_ds_percent)
        train_ds_len = train_ds_len - val_ds_len
        val_ds = train_ds.skip(train_ds_len).take(val_ds_len)
        train_ds = train_ds.take(train_ds_len)
    else:
        val_ds = None
    if reduce_train_set > 0:
        if skip_train_set + reduce_train_set > len(train_ds):
            raise ValueError("Requested more sample from the train set that available!")
        train_ds = train_ds.skip(skip_train_set).take(reduce_train_set)

    if reduce_val_set > 0 and val_ds is not None:
        if skip_val_set + reduce_val_set > len(val_ds):
            raise ValueError("Requested more sample from the val set that available!")
        val_ds = val_ds.skip(skip_val_set).take(reduce_val_set)

    train_ds = enable_for_ground_truth(train_ds, marker_last_frame=marker_last_frame)
    if val_ds is not None:
        val_ds = enable_for_ground_truth(val_ds, marker_last_frame=marker_last_frame)
    
    return train_ds, val_ds


def enable_for_ground_truth(dataset, prepend_n=0, append_n=0, marker_last_frame=False):
    """
    Add a column at the end of the dataset to indicate that the marker is enabled (i.e. 1).
    if prepend_n is > 0 also add some disabled markers at the beginning of the set of markers (for each frame)
    if append_n is > 0 also add some disabled markers at the end of the set of markers (for each frame).
    This function is necessary to use selective losses.
    marker_last_frame = True if you want to extract the last frame and use only that one for the computation of the loss
    """
    if dataset is None:
        return None

    if marker_last_frame:
        def extract_last_frame(in_tensor):
            in_tensor = in_tensor[-1]
            in_tensor = tf.expand_dims(in_tensor, 0) # restore frame axis
            return in_tensor
        dataset = dataset.map(lambda x: extract_last_frame(x))
    
    for t in dataset.take(1):
        last_shape_num = t.shape[-1]
        new_shape = (t.shape[:-1]+1) # take shape up to the last dimension and add 1 for the last column (+1 means one column, not 3+1)

    def get_01_vector(in_tensor):
        # get a tensor of 0 and 1, that is 0 when the corresponding row (only first column is considered) in in_tensor is NaN
        enabled_col = tf.cast(tf.math.logical_not(tf.math.is_nan(in_tensor[..., 0])), dtype=tf.float32)
        enabled_col = tf.expand_dims(enabled_col, -1)
        return tf.concat([in_tensor, enabled_col], axis=-1)

    dataset = dataset.map(lambda x: get_01_vector(x)) # map to add the enabled column

    if prepend_n > 0:
        # compute new shape as (old_shape[0], prepend_n+new_shape[1], 4) (this is how it should be, the code is generalized to support different shapes)
        new_shape = (new_shape[:-2]+(prepend_n+new_shape[-2])+(last_shape_num+1)) # xyz + enabled/disabled
        extended = tf.zeros(shape=new_shape) # set as 0 to disable (also the markers position is set to 0 for simplicity)
        dataset = dataset.map(lambda x: tf.concat([extended, x], axis=-2)) # append to the markers
        new_shape = (new_shape[:-1]+1) # restore 1 for last column
    if append_n > 0:
        # compute new shape as (old_shape[0], new_shape[1]+append_n, 4) (this is how it should be, the code is generalized to support different shapes)
        new_shape = (new_shape[:-2]+(new_shape[-2]+append_n)+(last_shape_num+1)) # xyz + enabled/disabled
        extended = tf.zeros(shape=new_shape) # set as 0 to disable (also the markers position is set to 0 for simplicity)
        dataset = dataset.map(lambda x: tf.concat([x, extended], axis=-2)) # append to the markers
        new_shape = (new_shape[:-1]+1) # restore 1 for last column
    return dataset


def convert_dataset(dataset_vars):
    """
    Simple utility function that I used at some point...
    Build and save a dataset at its final stage (before preprocessing for training).
    Note that the dataset saved is built for a specific number of time steps and stride.
    """
    print("------------------------------")
    print("\tParsing vars")
    dataset_vars['reduce_train_set'] = -1
    dataset_vars['reduce_val_set'] = -1
    # dataset
    print("------------------------------")
    print("\tPreparing dataset")
    split_train_ds, split_val_ds = setup_dataset(**dataset_vars)
    print("------------------------------")
    print("\tSaving to file")
    print("Training exp...", end="\t")
    save_dataset(dataset=split_train_ds, time_steps=dataset_vars['time_steps'], time_steps_stride=dataset_vars['time_steps_stride'], dataset_folder=dataset_vars['dataset_dir'], as_records=False, subfolder="train")
    print("Done")
    print("Training records...", end="\t")
    save_dataset(dataset=split_train_ds, time_steps=dataset_vars['time_steps'], time_steps_stride=dataset_vars['time_steps_stride'], dataset_folder=dataset_vars['dataset_dir'], as_records=True, subfolder="train")
    print("Done")
    print("Validation exp...", end="\t")
    save_dataset(dataset=split_val_ds, time_steps=dataset_vars['time_steps'], time_steps_stride=dataset_vars['time_steps_stride'], dataset_folder=dataset_vars['dataset_dir'], as_records=False, subfolder="val")
    print("Done")
    print("Validation records...", end="\t")
    save_dataset(dataset=split_val_ds, time_steps=dataset_vars['time_steps'], time_steps_stride=dataset_vars['time_steps_stride'], dataset_folder=dataset_vars['dataset_dir'], as_records=True, subfolder="val")
    print("Done")
