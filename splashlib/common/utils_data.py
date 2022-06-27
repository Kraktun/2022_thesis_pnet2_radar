import tensorflow as tf
import os
import numpy as np
from pathlib import Path
import pandas as pd
import shutil

# Note that we assume that a sequence has always less than 1000 frames (output files are hardcoded to have 3 digits).

def reduce_marker_set(original_markers, keep_markers, markers_dir):
    """
    Reduce the set of markers in each frame by keeping only those specified by the provided labels.
    Note that this is done IN PLACE, so make a copy of your dataset first.

    original_markers: ordered (same as csv files) list of labels for the markers
    keep_markers: list of labels of the markers to keep
    markers_dir: folder where the marker csv files are
    """
    # get indexes of markers to keep
    keep_indexes = [original_markers.index(m) for m in keep_markers]

    dirs = os.listdir(markers_dir)
    for d in dirs:
        folder_name = os.path.join(markers_dir, d)
        frames = Path(folder_name).glob("*.csv")
        frames = sorted(frames)
        for frame in frames:
            frame_data = np.loadtxt(frame, delimiter=',')
            frame_data = frame_data[keep_indexes]
            np.savetxt(frame, frame_data, delimiter=",", fmt='%1.14f')

def cut_sequences(dataset_dir, markers_dir, cut_file, out_dir, markers_out_dir, keep_empty_frame_n=0):
    """
    Reduce a sequence of frames according to the cut values computed by the matching algorithm (or the provided
    file if different). The function reads the cut_file and for each sequence in the file it copies only the frames 
    included in the range, left extreme included, right extreme excluded.

    dataset_dir: input dir of the dataset (should end with /train)
    markers_dir: folder with the markers (should end with /markers)
    cut_file: path to the cut file written by the matching algorithm, contains tuples with values
        (sequence_number, first_frame_number, last_frame_number)
    out_dir: path to the folder where to copy the cut sequences (should end with /train)
    markers_out_dir: path to the folder where to copy the cut markers
    keep_empty_frame_n: set to a int > 0 to keep a certain amount of empty marker frames at the beginning
        This allows to keep more information from the radar by appending empty frames to the marker sequence.
        If for instance you use 30 radar frame and only the last marker frame for the training, you can accept 29 empty 
        marker frames before your first real value.
    """
    df = pd.read_csv(cut_file, sep=',', header=None, dtype=str)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    cut_seqs = df.values
    # get the shape of a marker frame
    first_folder = [f for f in os.listdir(markers_dir) if os.path.isdir(os.path.join(markers_dir, f))][0]
    first_csv = next((Path(markers_dir) / first_folder).glob("*.csv"))
    # create a frame full of NaN
    # we don't copy directly from the marker sequence as we don't know if it's actually empty (we may want to
    # discard values for instance) nor if there are actually frames (radar sequence starts before the marker sequence).
    empty_frame = np.full_like(np.loadtxt(first_csv, delimiter=','), fill_value=np.nan)
    markers_subdirs = os.listdir(markers_dir)
    for dd in os.listdir(dataset_dir):
        # get range for current subfolder
        cut_tuple = [s for s in cut_seqs if dd.endswith(s[0])]
        if len(cut_tuple) > 0:
            print(f"Processing {dd}")
            from_idx = int(cut_tuple[0][1]) # [0] to select first (and only) result, [1] for left extreme included
            if keep_empty_frame_n > 0:
                # compute number of empty frames to add
                new_from_idx = max(from_idx-keep_empty_frame_n, 0)
                empty_frames_to_add = from_idx-new_from_idx
                from_idx = new_from_idx
            else:
                empty_frames_to_add = 0
            to_idx = int(cut_tuple[0][2]) # right extreme excluded
            # generate the range (note that the initial empty frames are not copied from the marker sequence)
            copy_range = range(from_idx, to_idx)
            # copy dataset frames
            out_path = Path(out_dir) / dd
            out_path.mkdir(exist_ok=True, parents=True)
            for c in (Path(dataset_dir) / dd).glob("*.csv"):
                if int(c.stem) in copy_range:
                    shutil.copy(c, out_path)
            # copy markers frames
            # get equivalent marker dir for the current dataset dir
            current_marker_subdir = [cur_folder for cur_folder in markers_subdirs if cur_folder.endswith(cut_tuple[0][0])][0]
            # build output path
            markers_out_path = Path(markers_out_dir) / current_marker_subdir
            markers_out_path.mkdir(exist_ok=True, parents=True)
            # copy all marker frames and prepend NaN frames if specified
            for i, frame_to_copy in enumerate(copy_range): 
                out_filename = f"{frame_to_copy:03d}.csv"
                out_file = os.path.join(markers_out_path, out_filename)
                if i < empty_frames_to_add:
                    # copy an empty frame
                    np.savetxt(out_file, empty_frame, delimiter=",", fmt='%1.1f')
                else:
                    # copy the correct marker frame
                    source_file = os.path.join(markers_dir, current_marker_subdir, out_filename)
                    shutil.copy(source_file, out_file)


def load_csv(folder_name):
    """
    Simple function to load the csv files inside a folder as a list of numpy arrays.
    """
    frames = Path(folder_name).glob("*.csv")
    frames = sorted(frames)
    frames = [np.loadtxt(f, delimiter=',') for f in frames]
    return frames

def refill_dataset(dataset_dir, crop_max=-1, replication_strategy="random", print_only=False):
    """
    Given a list of sequences for a capture, either find the number of points in each frame, 
    or add new points/remove points to reach a certain threshold.
    This is done IN PLACE. Make a copy of the dataset first.

    dataset_dir: input dir of the dataset (should end with /train)
    crop_max: set threshold number of points in each frame. Adds new points if a frame has less, or remove
        points if a frame has more. If not set (i.e. None or < 0) the function add points until global maximum is reached.
    replication_strategy: defines how to add new points if a frame has less points than crop_max. 
        Available options are: 
            'random' to sample at random from the set of already present points in the frame
            'last' to repeat the last point in the frame multiple times
            'inf' to add points with inf value
            'inf_zero' to add points with inf value for the first three columns (x,y,z) and 0 for the others (velocity etc.)
    print_only: set to True to find the number of points per frame. No modification is made to the data.
    """
    dirs = os.listdir(dataset_dir)

    frames_point_counter = []
    # collect some statistics on points, from 1 to 600
    for i in range(0, 600, 50):
        frames_point_counter.append((i, 0))

    max_global = 0

    for d in dirs:
        # each dir is a track of multiple frames
        filenames = tf.io.gfile.glob(os.path.join(dataset_dir, d, "*.csv"))
        if crop_max < 0 or print_only:
            # get max number of points in this sequence of frames
            for f in filenames:
                with open(f, "r") as fr:
                    file_length = len(fr.readlines()) 
                max_global = max(max_global, file_length)
                
                for i, c in enumerate(frames_point_counter):
                    if file_length > c[0]:
                        frames_point_counter[i] = (c[0], c[1]+1)
    
    if crop_max < 0 or print_only:
        print(f"\nGlobal max is {max_global}")
        for c in frames_point_counter:
            print(f"Frames with more than {c[0]} points are: {c[1]}")
    
    if print_only:
        return
    
    def fill_or_empty(filename, up_to, rng):
        """
        Add or remove points according to the main threshold defined in the main function.
        """
        with open(filename, "r") as fr:
            file_data = fr.readlines()
        file_length = len(file_data)
        if file_length < up_to:
            # add points
            if replication_strategy == "last":
                # take last point and repeat it
                file_data.extend(file_data[-1]*(up_to-file_length))
            elif replication_strategy == "random":
                # sample at random with replacement
                file_data.extend(rng.choice(file_data, size=(up_to-file_length), replace=True))
            elif replication_strategy == "inf":
                # add inf points
                file_data.extend([','.join([str(np.inf)]*len(file_data[-1].split(','))) + '\n']*(up_to-file_length))
            elif replication_strategy == "inf_zero":
                # add points with x,y,z set to inf and the other columns set to 0
                inf_list = [str(np.inf)]*3
                zero_list = ['0']*(len(file_data[-1].split(','))-3)
                inf_list.extend(zero_list)
                file_data.extend([','.join(inf_list) + '\n']*(up_to-file_length))
            elif replication_strategy == "zero":
                # add points with values 0
                file_data.extend([','.join(['0.0']*len(file_data[-1].split(','))) + '\n']*(up_to-file_length))
        elif file_length > up_to:
            # remove points by sampling at random without replacement
            file_data = rng.choice(file_data, size=up_to, replace=False)
        with open(filename, "w") as fr:
            fr.writelines(file_data)
    
    rng = np.random.default_rng()
    if crop_max > 0:
        up_to = crop_max
    else:
        up_to = max_global
    for d in dirs:
        filenames = tf.io.gfile.glob(os.path.join(dataset_dir, d, "*.csv"))
        for f in filenames:
            fill_or_empty(f, up_to, rng)

def reduce_by_limit(dataset_dir, xlim=None, ylim=None, zlim=None):
    """
    Reduce number of points in a frame to keep only those inside the limits.
    This is done IN PLACE. Make a copy of the dataset first.

    dataset_dir: input dir of the dataset (should end with /train)
    xlim: tuple (from, to) to set the x limit
    ylim: tuple (from, to) to set the y limit
    zlim: tuple (from, to) to set the z limit
    """
    dirs = os.listdir(dataset_dir)
    for d in dirs:
        filenames = tf.io.gfile.glob(os.path.join(dataset_dir, d, "*.csv"))
        for f in filenames:
            file_data = np.loadtxt(f, delimiter=',')
            x_s = file_data[:,0]
            y_s = file_data[:,1]
            z_s = file_data[:,2]
            # apply limits
            _, _, _, mask = constrict_ranges(x_s, y_s, z_s, xlim, ylim, zlim)
            file_data = file_data[mask]
            np.savetxt(f, file_data, delimiter=",", fmt='%1.14f') # max precision of the radar, goes up to 16 decimals only for the last columns
    

def discard_out_of_range_points(points, lim):
    # returns integers
    #ind = np.where((points > lim[0]) & (points < lim[1]))
    # returns true/false
    ind = np.logical_and(np.greater_equal(points, lim[0]), np.less_equal(points, lim[1]))
    return ind

def constrict_ranges(x_s, y_s, z_s, xlim=None, ylim=None, zlim=None):
    """
    Keep only the points that respect the limits defined, note that a point is defined as an element with the same index
    in all three vectors x_s, y_s, z_s, i.e. if one element at index i of x_s does not respect xlim, the element of y_s and z_s
    at index i will also be discarded. The final mask is also returned.
    """
    if xlim is not None:
        x_ind = discard_out_of_range_points(x_s, xlim)
    else:
        if not isinstance(x_s, np.ndarray):
            x_s = np.array([x_s])
        x_ind = [True] * x_s.size
    if ylim is not None:
        y_ind = discard_out_of_range_points(y_s, ylim)
    else:
        if not isinstance(y_s, np.ndarray):
            y_s = np.array([y_s])
        y_ind = [True] * y_s.size
    if zlim is not None:
        z_ind = discard_out_of_range_points(z_s, zlim)
    else:
        if not isinstance(z_s, np.ndarray):
            z_s = np.array([z_s])
        z_ind = [True] * z_s.size
    mask = np.logical_and(x_ind, y_ind, dtype=np.int64)
    mask = np.logical_and(mask, z_ind, dtype=np.int64)
    return x_s[mask], y_s[mask], z_s[mask], mask
