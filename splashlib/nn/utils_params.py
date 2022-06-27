import json
import argparse

"""
I won't comment this file as it's the repetition of the same thing over and over.
Just know that it parses all the CLI options and adds the default values if not provided.
Use -h for a list of the options.
"""

default_seed = 211
# dataset vars
default_dataset_dir = "../datasets/2021-12-17_deis/cl_256pc_random"
default_val_dir = ""
default_max_points = 256
default_global_feats = 9
# dataset-training vars
default_reduce_train_set = -1
default_skip_train_set = 0
default_reduce_val_set = -1
default_skip_val_set = 0
default_val_ds_from_train = True
default_val_ds_percent = 0.2
# structure vars
default_time_steps = 30
default_time_steps_stride = 30
default_marker_last_frame = False
# dataset data aug
default_dataset_shift_apply = False
default_dataset_shift_append = False
default_dataset_shift_prob = 0.3
default_dataset_shift_interval = [-1,1]
default_dataset_center_seq = False
default_dataset_center_seq_mode = "all"
default_dataset_center_seq_dims = None
default_dataset_shuffle = False
default_dataset_normalization_apply = False
default_dataset_normalization_axis = [0,1,2]
default_dataset_normalization_xyz = False
default_dataset_normalization_sequence = [False,False,False]
default_dataset_normalize_after_center_seq = True
default_dataset_rotation_apply = False
default_dataset_rotation_append = False
default_dataset_rotation_prob = 0.3
default_dataset_rotation_intervals = [-1,1]*3

# preprocessing vars
default_preproc_prefix = "local"
default_preproc_name = "NEW_DS_BASE_1"
default_preproc_type = "generic"
default_preproc_out_time_steps = None
default_preproc_out_time_steps_stride = None
default_preproc_standardize = False
default_preproc_skip_standardization_axis = []
default_preproc_eps = 1e-6
default_preproc_keep_features = [0,1,2]
default_preproc_add_channel = False
default_preproc_random_shift = False
default_preproc_random_shift_prob = 0.3
default_preproc_random_shift_interval = [-1,1]
default_preproc_random_perm = False
default_preproc_random_perm_prob = 0.3
default_preproc_random_perm_axis = [0,1,2]
default_preproc_random_shuffle = False
default_preproc_random_shuffle_prob = 0.3
default_preproc_random_rotation = False
default_preproc_random_rotation_prob = 0.3
default_preproc_random_rotation_angles = [[-1,1]]*3

# model vars
default_model_prefix = "cluster"
default_model_name = "P-NET2_1"
default_model_type = "autoencoder"
default_model_enc_type = "pointnet2"
default_model_enc_args = "pointnet2_sem_seg_enc_reduced"
default_model_rnn_type = "gru"
default_model_rnn_args = "gru_baseline_copy"
default_model_dec_type = "pointnet2"
default_model_dec_args = "pointnet2_sem_seg_dec_reduced"
default_model_loss = "chamfer"
default_model_optimizer = "adam"

# learning
default_train_batch_size = 16
default_train_epochs = 50
default_train_load_from_checkpoint = False
default_train_checkpoint_mode = "best"

# save\load model
default_save_model_name = "" # empty to use default naming convention


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with configurable parameters')
    parser.add_argument('--seed', help='seed for random libraries', default=default_seed, type=int)
    # input files that include all parameters
    parser.add_argument('--dataset-input', help='Input json for the dataset variables', type=argparse.FileType('r'))
    parser.add_argument('--preproc-input', help='Input json for the preprocessing variables', type=argparse.FileType('r'))
    parser.add_argument('--model-input', help='Input json for the model variables', type=argparse.FileType('r'))
    parser.add_argument('--train-input', help='Input json for the training variables', type=argparse.FileType('r'))
    # single variables 
    # dataset vars
    parser.add_argument('--dataset-dir', help='Directory of the training dataset', type=str)
    parser.add_argument('--val-dir', help='Directory of the validation dataset', type=str)
    parser.add_argument('--num-subjects', help='Number of subjects in the dataset', type=int)
    parser.add_argument('--max-points', help='Maximum number of points in each frame', type=int)
    parser.add_argument('--global-feats', help='Number of features in the dataset', type=int)
    parser.add_argument('--reduce-train-set', help='Number of tracks to keep from the train dataset', type=int)
    parser.add_argument('--skip-train-set', help='Number of samples to skip before keeping tracks in the train dataset', type=int)
    parser.add_argument('--reduce-val-set', help='Number of tracks to keep from the val dataset', type=int)
    parser.add_argument('--skip-val-set', help='Number of samples to skip before keeping tracks in the val dataset', type=int)
    parser.add_argument('--val-ds-from-train', help='Extract val ds from train ds', action='store_true', default=None)
    parser.add_argument('--val-ds-percent', help='Percentage of the train set to use for validation', type=float)
    parser.add_argument('--time-steps', help='Number of frames to collect for a single track (i.e. a sequence in the RNN)', type=int)
    parser.add_argument('--time-steps-stride', help='Number of frames from one track to the following one (a value equal to or bigger than time-steps means no overlapping)', type=int)
    parser.add_argument('--marker-last-frame', help='Use only last frame for the markers to compute the loss', action='store_true', default=None)
    parser.add_argument('--dataset-shift', help='Apply a random shift on the whole dataset (markers included)', dest='dataset_shift_apply', action='store_true', default=None)
    parser.add_argument('--no-dataset-shift', help='Do not apply a random shift on the whole dataset (markers included)', dest='dataset_shift_apply', action='store_false', default=None)
    parser.add_argument('--dataset-shift-append', help='Append shifted dataset after the unshifted one', action='store_true', default=None)
    parser.add_argument('--dataset-shift-prob', help='Define probability of a random shift of the dataset', type=float)
    parser.add_argument('--dataset-shift-interval', help='Define the range of the random shift of the dataset', type=list)
    parser.add_argument('--dataset-center-seq', help='Center sequences to 0 according to provided mode', dest='dataset_center_seq', action='store_true', default=None)
    parser.add_argument('--dataset-no-center-seq', help='Do not center sequences to 0 according to provided mode', dest='dataset_center_seq', action='store_false', default=None)
    parser.add_argument('--dataset-center-seq-mode', help='Define mode to center sequences among [all, first, middle, last]', type=str)
    parser.add_argument('--dataset-center-seq-dims', help='Define feature axis to center in 0 as a list', type=list)
    parser.add_argument('--dataset-shuffle', help='Shuffle the dataset', dest='dataset_shuffle', action='store_true', default=None)
    parser.add_argument('--dataset-no-shuffle', help='Don\'t shuffle the dataset', dest='dataset_shuffle', action='store_false', default=None)
    parser.add_argument('--dataset-normalization', help='Normalize dataset to [0-1]', dest='dataset_normalization_apply', action='store_true', default=None)
    parser.add_argument('--dataset-no-normalization', help='Do not normalize dataset to [0-1]', dest='dataset_normalization_apply', action='store_false', default=None)
    parser.add_argument('--dataset-normalization-axis', help='Define axis where to apply normalization', type=list)
    parser.add_argument('--dataset-normalization-xyz', help='Set to normalize xyz together', action='store_true', default=None)
    parser.add_argument('--dataset-normalization-sequence', help='Set to normalize all the sequence together', type=list)
    parser.add_argument('--dataset-normalize-after-center-seq', help='Normalize dataset after sequence centering', dest='dataset_normalize_after_center_seq', action='store_true', default=None)
    parser.add_argument('--dataset-rotation', help='Apply a random rotation on the whole dataset (markers included)', dest='dataset_rotation_apply', action='store_true', default=None)
    parser.add_argument('--no-dataset-rotation', help='Do not apply a random rotation on the whole dataset (markers included)', dest='dataset_rotation_apply', action='store_false', default=None)
    parser.add_argument('--dataset-rotation-append', help='Append rotated dataset after the unrotated one', action='store_true', default=None)
    parser.add_argument('--dataset-rotation-prob', help='Define probability of a random rotation of the dataset', type=float)
    parser.add_argument('--dataset-rotation-intervals', help='Define the range of the random rotation of the dataset', type=list)
    # preproc variables
    parser.add_argument('--preproc-prefix', help='Prefix to append to the name of the preprocessing model. Ignored if preproc-name is specified', type=str)
    parser.add_argument('--preproc-name', help='Name of the preprocessing model, if not specified it\'s automatically generated', type=str)
    parser.add_argument('--preproc-type', help='Type of the preprocessing model', type=str)
    parser.add_argument('--preproc-out-time-steps', help='Splits the tracks from the dataset in smaller subtracks. Must be smaller than time-steps. If not defined, keeps original tracks. [EXPERIMENTAL]', type=int)
    parser.add_argument('--preproc-out-time-steps-stride', help='Number of frames between two subsequent tracks for the new subtracks. Ignored if preproc-out-time-steps is not valid. [EXPERIMENTAL]', type=int)
    parser.add_argument('--preproc-standardize', help='Apply standardization. [EXPERIMENTAL]', action='store_true', default=None)
    parser.add_argument('--preproc-no-standardize', help='Do not apply standardization. [EXPERIMENTAL]', dest='preproc_standardize', action='store_false', default=None)
    parser.add_argument('--preproc-skip-standardization-axis', help='Subset of axis where not to apply standardization. Applied after feature reduction. [EXPERIMENTAL]', type=list)
    parser.add_argument('--preproc-eps', help='Epsilon used for standardization', type=float)
    parser.add_argument('--preproc-keep-features', help='Keep only a subset of features from the input dataset', type=list)
    parser.add_argument('--preproc-add-channel', help='Add the channel dimension to the dataset', action='store_true', default=None)
    parser.add_argument('--preproc-no-add-channel', help='Do not add the channel dimension to the dataset', dest='preproc_add_channel', action='store_false', default=None)
    parser.add_argument('--preproc-random-shift', help='Apply a random shift of the input points during training', dest='preproc_random_shift', action='store_true', default=None)
    parser.add_argument('--preproc-no-random-shift', help='Do not apply a random shift of the input points during training', dest='preproc_random_shift', action='store_false', default=None)
    parser.add_argument('--preproc-random-shift-prob', help='Define probability of a random shift of the input points during training', type=float)
    parser.add_argument('--preproc-random-shift-interval', help='Define the range of the random shift of the input points during training', type=list)
    parser.add_argument('--preproc-random-perm', help='Apply a random permutation of the input points during training. [EXPERIMENTAL]', dest='preproc_random_perm', action='store_true', default=None)
    parser.add_argument('--preproc-no-random-perm', help='Do not apply a random permutation of the input points during training. [EXPERIMENTAL]', dest='preproc_random_perm', action='store_false', default=None)
    parser.add_argument('--preproc-random-perm-prob', help='Define probability of a random permutation of the input points during training. [EXPERIMENTAL]', type=float)
    parser.add_argument('--preproc-random-perm-axis', help='Define the axis of the random permutation of the input points during training. [EXPERIMENTAL]', type=list)
    parser.add_argument('--preproc-random-shuffle', help='Apply a random shuffle of the points of each frame during training.', dest='preproc_random_shuffle', action='store_true', default=None)
    parser.add_argument('--preproc-no-random-shuffle', help='Do not apply a random shuffle of the input points during training.', dest='preproc_random_shuffle', action='store_false', default=None)
    parser.add_argument('--preproc-random-shuffle-prob', help='Define probability of a random shuffle of the input points during training.', type=float)
    parser.add_argument('--preproc-random-rotation', help='Apply a random rotation of the input points during training. [EXPERIMENTAL]', dest='preproc_random_rotation', action='store_true', default=None)
    parser.add_argument('--preproc-no-random-rotation', help='Do not apply a random rotation of the input points during training. [EXPERIMENTAL]', dest='preproc_random_rotation', action='store_false', default=None)
    parser.add_argument('--preproc-random-rotation-prob', help='Define probability of a random rotation of the input points during training. [EXPERIMENTAL]', type=float)
    parser.add_argument('--preproc-random-rotation-angles', help='Define the axis of the random rotation of the input points during training. [EXPERIMENTAL]', type=list)
    # model variables
    parser.add_argument('--model-prefix', help='Prefix to append to the name of the model. Ignored if model-name is specified', type=str)
    parser.add_argument('--model-name', help='Name of the model, if not specified it\'s automatically generated', type=str)
    parser.add_argument('--model-type', help='Type of the model', type=str)
    parser.add_argument('--model-enc-type', help='Type of the encoder', type=str)
    parser.add_argument('--model-enc-args', help='Subtype of the encoder (i.e. dictionary with the parameters)', type=str)
    parser.add_argument('--model-rnn-type', help='Type of the RNN', type=str)
    parser.add_argument('--model-rnn-args', help='Subtype of the RNN (i.e. dictionary with the parameters)', type=str)
    parser.add_argument('--model-dec-type', help='Type of the decoder', type=str)
    parser.add_argument('--model-dec-args', help='Subtype of the decoder (i.e. dictionary with the parameters)', type=str)
    parser.add_argument('--model-loss', help='Loss for the training', type=str)
    parser.add_argument('--model-optimizer', help='Optimizer for the training', type=str)
    # training variables
    parser.add_argument('--train-batch-size', '-batch', help='Batch size during training', type=int)
    parser.add_argument('--train-epochs', '-epochs', help='Number of epochs to train for', type=int)
    parser.add_argument('--train-load-from-checkpoint', help='Load weights from checkpoint rather than full model', dest='train_load_from_checkpoint', action='store_true', default=None)
    parser.add_argument('--train-checkpoint-mode', help='Type of checkpoint to load', type=str)
    # note: currently no 'callbacks' command line options
    # load-save variables
    parser.add_argument('--save-model-name', help='Output name of the learned weights. If not set default naming convention will be used', type=str)
    parser.add_argument('--load-model-name', help='Path to the weights to load to continue training. If not set training will start from scratch. Specify as .h5 file or a folder/* to choose automatically the correct weights based on the name', type=str)

    args = parser.parse_args()
    dataset_vars = load_json(args.dataset_input)
    preproc_vars = load_json(args.preproc_input)
    model_vars = load_json(args.model_input)
    train_vars = load_json(args.train_input)

    # overwrite vars
    if args.dataset_dir:
        dataset_vars['dataset_dir'] = args.dataset_dir
    if args.val_dir:
        dataset_vars['val_dir'] = args.val_dir
    if args.max_points:
        dataset_vars['max_points'] = args.max_points
    if args.global_feats:
        dataset_vars['global_feats'] = args.global_feats
    if args.reduce_train_set:
        dataset_vars['reduce_train_set'] = args.reduce_train_set
    if args.skip_train_set:
        dataset_vars['skip_train_set'] = args.skip_train_set
    if args.reduce_val_set:
        dataset_vars['reduce_val_set'] = args.reduce_val_set
    if args.skip_val_set:
        dataset_vars['skip_val_set'] = args.skip_val_set
    if args.val_ds_from_train is not None: # explicit as we need to access it if it's false
        dataset_vars['val_ds_from_train'] = args.val_ds_from_train
    if args.val_ds_percent:
        dataset_vars['val_ds_percent'] = args.val_ds_percent
    if args.time_steps:
        dataset_vars['time_steps'] = args.time_steps
    if args.time_steps_stride:
        dataset_vars['time_steps_stride'] = args.time_steps_stride
    if args.marker_last_frame is not None:
        dataset_vars['marker_last_frame'] = args.marker_last_frame
    if args.dataset_shift_apply is not None:
        dataset_vars['dataset_shift_apply'] = args.dataset_shift_apply
    if args.dataset_shift_append is not None:
        dataset_vars['dataset_shift_append'] = args.dataset_shift_append
    if args.dataset_shift_prob:
        dataset_vars['dataset_shift_prob'] = args.dataset_shift_prob
    if args.dataset_shift_interval:
        dataset_vars['dataset_shift_interval'] = args.dataset_shift_interval
    if args.dataset_center_seq is not None:
        dataset_vars['dataset_center_seq'] = args.dataset_center_seq
    if args.dataset_center_seq_mode:
        dataset_vars['dataset_center_seq_mode'] = args.dataset_center_seq_mode
    if args.dataset_center_seq_dims:
        dataset_vars['dataset_center_seq_dims'] = args.dataset_center_seq_dims
    if args.dataset_shuffle is not None:
        dataset_vars['dataset_shuffle'] = args.dataset_shuffle
    if args.dataset_normalization_apply is not None:
        dataset_vars['dataset_normalization_apply'] = args.dataset_normalization_apply
    if args.dataset_normalization_axis:
        dataset_vars['dataset_normalization_axis'] = args.dataset_normalization_axis
    if args.dataset_normalization_xyz is not None:
        dataset_vars['dataset_normalization_xyz'] = args.dataset_normalization_xyz
    if args.dataset_normalization_sequence:
        dataset_vars['dataset_normalization_sequence'] = args.dataset_normalization_sequence
    if args.dataset_normalize_after_center_seq is not None:
        dataset_vars['dataset_normalize_after_center_seq'] = args.dataset_normalize_after_center_seq
    if args.dataset_rotation_apply is not None:
        dataset_vars['dataset_rotation_apply'] = args.dataset_rotation_apply
    if args.dataset_rotation_append is not None:
        dataset_vars['dataset_rotation_append'] = args.dataset_rotation_append
    if args.dataset_rotation_prob:
        dataset_vars['dataset_rotation_prob'] = args.dataset_rotation_prob
    if args.dataset_rotation_intervals:
        dataset_vars['dataset_rotation_intervals'] = args.dataset_rotation_intervals
    
    if args.preproc_prefix:
        preproc_vars['preproc_prefix'] = args.preproc_prefix
    if args.preproc_name:
        preproc_vars['preproc_name'] = args.preproc_name
    if args.preproc_type:
        preproc_vars['preproc_type'] = args.preproc_type
    if args.preproc_out_time_steps:
        preproc_vars['preproc_out_time_steps'] = args.preproc_out_time_steps
    if args.preproc_out_time_steps_stride:
        preproc_vars['preproc_out_time_steps_stride'] = args.preproc_out_time_steps_stride
    if args.preproc_standardize is not None:
        preproc_vars['preproc_standardize'] = args.preproc_standardize
    if args.preproc_skip_standardization_axis:
        preproc_vars['preproc_skip_standardization_axis'] = args.preproc_skip_standardization_axis
    if args.preproc_eps:
        preproc_vars['preproc_eps'] = args.preproc_eps
    if args.preproc_keep_features:
        preproc_vars['preproc_keep_features'] = args.preproc_keep_features
    if args.preproc_add_channel is not None:
        preproc_vars['preproc_add_channel'] = args.preproc_add_channel
    if args.preproc_random_shift is not None:
        preproc_vars['preproc_random_shift'] = args.preproc_random_shift
    if args.preproc_random_shift_prob:
        preproc_vars['preproc_random_shift_prob'] = args.preproc_random_shift_prob
    if args.preproc_random_shift_interval:
        preproc_vars['preproc_random_shift_interval'] = args.preproc_random_shift_interval
    if args.preproc_random_perm is not None:
        preproc_vars['preproc_random_perm'] = args.preproc_random_perm
    if args.preproc_random_perm_prob:
        preproc_vars['preproc_random_perm_prob'] = args.preproc_random_perm_prob
    if args.preproc_random_perm_axis:
        preproc_vars['preproc_random_perm_axis'] = args.preproc_random_perm_axis
    if args.preproc_random_shuffle is not None:
        preproc_vars['preproc_random_shuffle'] = args.preproc_random_shuffle
    if args.preproc_random_shuffle_prob:
        preproc_vars['preproc_random_shuffle_prob'] = args.preproc_random_shuffle_prob
    if args.preproc_random_rotation is not None:
        preproc_vars['preproc_random_rotation'] = args.preproc_random_rotation
    if args.preproc_random_rotation_prob:
        preproc_vars['preproc_random_rotation_prob'] = args.preproc_random_rotation_prob
    if args.preproc_random_rotation_angles:
        preproc_vars['preproc_random_rotation_angles'] = args.preproc_random_rotation_angles
    
    if args.model_prefix:
        model_vars['model_prefix'] = args.model_prefix
    if args.model_name:
        model_vars['model_name'] = args.model_name
    if args.model_type:
        model_vars['model_type'] = args.model_type
    if args.model_enc_type:
        model_vars['model_enc_type'] = args.model_enc_type
    if args.model_enc_args:
        model_vars['model_enc_args'] = args.model_enc_args
    if args.model_rnn_type:
        model_vars['model_rnn_type'] = args.model_rnn_type
    if args.model_rnn_args:
        model_vars['model_rnn_args'] = args.model_rnn_args
    if args.model_dec_type:
        model_vars['model_dec_type'] = args.model_dec_type
    if args.model_dec_args:
        model_vars['model_dec_args'] = args.model_dec_args
    if args.model_loss:
        model_vars['model_loss'] = args.model_loss
    if args.model_optimizer:
        model_vars['model_optimizer'] = args.model_optimizer
    
    if args.train_batch_size:
        train_vars['train_batch_size'] = args.train_batch_size
    if args.train_epochs:
        train_vars['train_epochs'] = args.train_epochs
    if args.train_load_from_checkpoint is not None:
        train_vars['load_from_checkpoint'] = args.train_load_from_checkpoint
    if args.train_checkpoint_mode:
        train_vars['checkpoint_mode'] = args.train_checkpoint_mode
    
    
    seed = args.seed
    save_model_name = args.save_model_name
    load_model_name = args.load_model_name

    # load defaults where not available from json nor cli
    if 'dataset_dir' not in dataset_vars.keys():
        dataset_vars['dataset_dir'] = default_dataset_dir
    if 'val_dir' not in dataset_vars.keys():
        dataset_vars['val_dir'] = default_val_dir
    if 'max_points' not in dataset_vars.keys():
        dataset_vars['max_points'] = default_max_points
    if 'global_feats' not in dataset_vars.keys():
        dataset_vars['global_feats'] = default_global_feats
    if 'reduce_train_set' not in dataset_vars.keys():
        dataset_vars['reduce_train_set'] = default_reduce_train_set
    if 'skip_train_set' not in dataset_vars.keys():
        dataset_vars['skip_train_set'] = default_skip_train_set
    if 'reduce_val_set' not in dataset_vars.keys():
        dataset_vars['reduce_val_set'] = default_reduce_val_set
    if 'skip_val_set' not in dataset_vars.keys():
        dataset_vars['skip_val_set'] = default_skip_val_set
    if 'val_ds_from_train' not in dataset_vars.keys():
        dataset_vars['val_ds_from_train'] = default_val_ds_from_train
    if 'val_ds_percent' not in dataset_vars.keys():
        dataset_vars['val_ds_percent'] = default_val_ds_percent
    if 'time_steps' not in dataset_vars.keys():
        dataset_vars['time_steps'] = default_time_steps
    if 'time_steps_stride' not in dataset_vars.keys():
        dataset_vars['time_steps_stride'] = default_time_steps_stride
    if 'marker_last_frame' not in dataset_vars.keys():
        dataset_vars['marker_last_frame'] = default_marker_last_frame
    if 'dataset_shift_apply' not in dataset_vars.keys():
        dataset_vars['dataset_shift_apply'] = default_dataset_shift_apply
    if 'dataset_shift_append' not in dataset_vars.keys():
        dataset_vars['dataset_shift_append'] = default_dataset_shift_append
    if 'dataset_shift_prob' not in dataset_vars.keys():
        dataset_vars['dataset_shift_prob'] = default_dataset_shift_prob
    if 'dataset_shift_interval' not in dataset_vars.keys():
        dataset_vars['dataset_shift_interval'] = default_dataset_shift_interval
    if 'dataset_center_seq' not in dataset_vars.keys():
        dataset_vars['dataset_center_seq'] = default_dataset_center_seq
    if 'dataset_center_seq_mode' not in dataset_vars.keys():
        dataset_vars['dataset_center_seq_mode'] = default_dataset_center_seq_mode
    if 'dataset_center_seq_dims' not in dataset_vars.keys():
        dataset_vars['dataset_center_seq_dims'] = default_dataset_center_seq_dims
    if 'dataset_shuffle' not in dataset_vars.keys():
        dataset_vars['dataset_shuffle'] = default_dataset_shuffle
    if 'dataset_normalization_apply' not in dataset_vars.keys():
        dataset_vars['dataset_normalization_apply'] = default_dataset_normalization_apply
    if 'dataset_normalization_xyz' not in dataset_vars.keys():
        dataset_vars['dataset_normalization_xyz'] = default_dataset_normalization_xyz
    if 'dataset_normalization_axis' not in dataset_vars.keys():
        dataset_vars['dataset_normalization_axis'] = default_dataset_normalization_axis
    if 'dataset_normalization_sequence' not in dataset_vars.keys():
        dataset_vars['dataset_normalization_sequence'] = default_dataset_normalization_sequence
    if 'dataset_normalize_after_center_seq' not in dataset_vars.keys():
        dataset_vars['dataset_normalize_after_center_seq'] = default_dataset_normalize_after_center_seq
    if 'dataset_rotation_apply' not in dataset_vars.keys():
        dataset_vars['dataset_rotation_apply'] = default_dataset_rotation_apply
    if 'dataset_rotation_append' not in dataset_vars.keys():
        dataset_vars['dataset_rotation_append'] = default_dataset_rotation_append
    if 'dataset_rotation_prob' not in dataset_vars.keys():
        dataset_vars['dataset_rotation_prob'] = default_dataset_rotation_prob
    if 'dataset_rotation_intervals' not in dataset_vars.keys():
        dataset_vars['dataset_rotation_intervals'] = default_dataset_rotation_intervals
    
    if 'preproc_prefix' not in preproc_vars.keys():
        preproc_vars['preproc_prefix'] = default_preproc_prefix
    if 'preproc_name' not in preproc_vars.keys():
        preproc_vars['preproc_name'] = default_preproc_name
    if 'preproc_type' not in preproc_vars.keys():
        preproc_vars['preproc_type'] = default_preproc_type
    if 'preproc_out_time_steps' not in preproc_vars.keys():
        preproc_vars['preproc_out_time_steps'] = default_preproc_out_time_steps
    if 'preproc_out_time_steps_stride' not in preproc_vars.keys():
        preproc_vars['preproc_out_time_steps_stride'] = default_preproc_out_time_steps_stride
    if 'preproc_standardize' not in preproc_vars.keys():
        preproc_vars['preproc_standardize'] = default_preproc_standardize
    if 'preproc_skip_standardization_axis' not in preproc_vars.keys():
        preproc_vars['preproc_skip_standardization_axis'] = default_preproc_skip_standardization_axis
    if 'preproc_eps' not in preproc_vars.keys():
        preproc_vars['preproc_eps'] = default_preproc_eps
    if 'preproc_keep_features' not in preproc_vars.keys():
        preproc_vars['preproc_keep_features'] = default_preproc_keep_features
    if 'preproc_add_channel' not in preproc_vars.keys():
        preproc_vars['preproc_add_channel'] = default_preproc_add_channel
    if 'preproc_random_shift' not in preproc_vars.keys():
        preproc_vars['preproc_random_shift'] = default_preproc_random_shift
    if 'preproc_random_shift_prob' not in preproc_vars.keys():
        preproc_vars['preproc_random_shift_prob'] = default_preproc_random_shift_prob
    if 'preproc_random_shift_interval' not in preproc_vars.keys():
        preproc_vars['preproc_random_shift_interval'] = default_preproc_random_shift_interval
    if 'preproc_random_perm' not in preproc_vars.keys():
        preproc_vars['preproc_random_perm'] = default_preproc_random_perm
    if 'preproc_random_perm_prob' not in preproc_vars.keys():
        preproc_vars['preproc_random_perm_prob'] = default_preproc_random_perm_prob
    if 'preproc_random_perm_axis' not in preproc_vars.keys():
        preproc_vars['preproc_random_perm_axis'] = default_preproc_random_perm_axis
    if 'preproc_random_shuffle' not in preproc_vars.keys():
        preproc_vars['preproc_random_shuffle'] = default_preproc_random_shuffle
    if 'preproc_random_shuffle_prob' not in preproc_vars.keys():
        preproc_vars['preproc_random_shuffle_prob'] = default_preproc_random_shuffle_prob
    if 'preproc_random_rotation' not in preproc_vars.keys():
        preproc_vars['preproc_random_rotation'] = default_preproc_random_rotation
    if 'preproc_random_rotation_prob' not in preproc_vars.keys():
        preproc_vars['preproc_random_rotation_prob'] = default_preproc_random_rotation_prob
    if 'preproc_random_rotation_angles' not in preproc_vars.keys():
        preproc_vars['preproc_random_rotation_angles'] = default_preproc_random_rotation_angles
    
    if 'model_prefix' not in model_vars.keys():
        model_vars['model_prefix'] = default_model_prefix
    if 'model_name' not in model_vars.keys():
        model_vars['model_name'] = default_model_name
    if 'model_type' not in model_vars.keys():
        model_vars['model_type'] = default_model_type
    if 'model_enc_type' not in model_vars.keys():
        model_vars['model_enc_type'] = default_model_enc_type
    if 'model_enc_args' not in model_vars.keys():
        model_vars['model_enc_args'] = default_model_enc_args
    if 'model_rnn_type' not in model_vars.keys():
        model_vars['model_rnn_type'] = default_model_rnn_type
    if 'model_rnn_args' not in model_vars.keys():
        model_vars['model_rnn_args'] = default_model_rnn_args
    if 'model_dec_type' not in model_vars.keys():
        model_vars['model_dec_type'] = default_model_dec_type
    if 'model_dec_args' not in model_vars.keys():
        model_vars['model_dec_args'] = default_model_dec_args
    if 'model_loss' not in model_vars.keys():
        model_vars['model_loss'] = default_model_loss
    if 'model_optimizer' not in model_vars.keys():
        model_vars['model_optimizer'] = default_model_optimizer
    
    if 'train_batch_size' not in train_vars.keys():
        train_vars['train_batch_size'] = default_train_batch_size
    if 'train_epochs' not in train_vars.keys():
        train_vars['train_epochs'] = default_train_epochs
    if 'load_from_checkpoint' not in train_vars.keys():
        train_vars['load_from_checkpoint'] = default_train_load_from_checkpoint
    if 'checkpoint_mode' not in train_vars.keys():
        train_vars['checkpoint_mode'] = default_train_checkpoint_mode
    
    default_load_model_name = f"notebook_dumps/models_in/{preproc_vars['preproc_name']}_{model_vars['model_name']}/*" # specify a .h5 file or use a * inside a folder to choose 

    if not seed:
        seed = default_seed
    if not save_model_name:
        save_model_name = default_save_model_name
    if not load_model_name:
        load_model_name = default_load_model_name    
    
    return seed, dataset_vars, preproc_vars, model_vars, train_vars, save_model_name, load_model_name


def load_json(arg):
    if arg:
        return json.load(arg)
    return {}

if __name__ == "__main__":
    _, d, p, m, t, _, _ = parse_args()
    print(d)