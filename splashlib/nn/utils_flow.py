import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import tensorflow as tf

from splashlib.nn.custom_callbacks import *
from splashlib.nn.PreprocModelBuilderK import PreprocModelBuilderK
from splashlib.nn import models_dict
from splashlib.nn.ModelBuilderK import ModelBuilderK
from splashlib.nn.k_utils import isNoneOrBlank
from splashlib.nn.nets.pointnet2 import pointnet2
from splashlib.nn import keras_layers
from splashlib.nn.data_map_funcs import apply_dataset_shift, apply_center_sequence, apply_dataset_normalization, apply_dataset_rotation

def setup_env(current_path, save_model_dir, seed, deterministic=False):
    # setup initial vars for the training
    print(f"Start at: {datetime.now()}")
    # change to dir where the file is located
    os.chdir(current_path)
    # set seeds
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # tf.keras.utils.set_random_seed(seed) # same as those above, but for tf nightly, waiting for it to be mainstream
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        # tf.config.experimental.enable_op_determinism() # same as above, but for tf nightly, waiting for it to be mainstream
    # prepare output dirs
    Path(save_model_dir).mkdir(parents=True, exist_ok=True)


def get_preprocessing_model(time_steps, max_points, global_feats, preproc_name, preproc_prefix, preproc_type, 
        preproc_out_time_steps, preproc_out_time_steps_stride, preproc_standardize, preproc_skip_standardization_axis, preproc_eps,
        preproc_keep_features, preproc_add_channel, preproc_random_shift, preproc_random_shift_prob, preproc_random_shift_interval,
        preproc_random_perm, preproc_random_perm_prob, preproc_random_perm_axis, preproc_random_shuffle, preproc_random_shuffle_prob,
        preproc_random_rotation, preproc_random_rotation_prob, preproc_random_rotation_angles):
    """
    Yes, it's ugly AF. Simply because at some point I changed the names and did not propagate the changes to the whole code.
    Feel free to do it, then you'll just need to use **params.
    """
    # build the preprocessing model
    my_preproc_model_name = preproc_name
    if isNoneOrBlank(my_preproc_model_name):
        my_preproc_model_name = f"{preproc_prefix}_{time_steps}-steps"
    return PreprocModelBuilderK(
        model_name=my_preproc_model_name, 
        model_type=preproc_type,
        points_in_frame=max_points, 
        num_features=global_feats,
        in_time_steps=time_steps, 
        out_time_steps=preproc_out_time_steps,
        out_time_steps_stride=preproc_out_time_steps_stride,
        standardize=preproc_standardize,
        skip_standardization_axis=preproc_skip_standardization_axis,
        eps=preproc_eps,
        keep_features=preproc_keep_features,
        add_channel=preproc_add_channel,
        random_shift=preproc_random_shift, 
        random_shift_prob=preproc_random_shift_prob, 
        random_shift_interval=preproc_random_shift_interval, 
        random_permutation=preproc_random_perm, 
        random_permutation_prob=preproc_random_perm_prob, 
        random_permutation_axis=preproc_random_perm_axis,
        random_shuffle=preproc_random_shuffle,
        random_shuffle_prob=preproc_random_shuffle_prob,
        random_rotation=preproc_random_rotation,
        random_rotation_prob=preproc_random_rotation_prob,
        random_rotation_angles=preproc_random_rotation_angles
    )


def get_model(model_prefix, model_name, model_type, model_enc_type, model_enc_args, model_rnn_type, 
        model_rnn_args, model_dec_type, model_dec_args, model_loss, model_optimizer):
    # build the learning model
    # args can be either strings (from predefined models_dict or custom models)
    if type(model_enc_args) == str:
        enc_args = models_dict.all_models[model_enc_args].copy()
    else:
        enc_args = model_enc_args.copy()

    if type(model_rnn_args) == str:
        rnn_args = models_dict.all_models[model_rnn_args].copy()
    else:
        rnn_args = model_rnn_args.copy()

    if type(model_dec_args) == str:
        dec_args = models_dict.all_models[model_dec_args].copy()
    else:
        dec_args = model_dec_args.copy()

    model_builder = ModelBuilderK(
        model_name=model_name, # updated later if None
        model_type=model_type,
        encoder_type=model_enc_type, 
        rnn_type=model_rnn_type, 
        decoder_type=model_dec_type,
        encoder_args=enc_args, 
        rnn_args=rnn_args, 
        decoder_args=dec_args,
        loss=model_loss,
        optimizer=model_optimizer
    )

    if isNoneOrBlank(model_builder.model_name):
        new_name = f"{model_prefix}_{model_builder.model_type}_{model_builder.encoder_type}_{model_builder.rnn_type}_{model_builder.decoder_type}"
        model_builder.model_name = new_name.replace(", ", "+") # replace commas in arrays values with concatenated +

    return model_builder


def load_saved(preproc_model_builder, model_builder, full_model, sub_models, load_model_name):
    """
    Load weights from a previous training session.
    This function is a bit of a mess as I changed it many times to match the evolution of the model 
    and currently is a bit tricky to understand.
    What it does in principle is to take a model already built but untrained, assume that the weights (saved as a complete model)
    are equal, load the weights and the corresponding model and split it in the different submodels (preproc, learning etc.)
    to match the untrained model.

    In practice however this was a mess, so currently I save only the weights (not the model) and load only the weights.
    The submodels should be updated with the trained weights accordingly.

    Loss and training parameters are taken from the model_builder.

    Currently models with multiple outputs are not supported for the division in blocks. So only the full model is returned updated.

    The function returns None whenever an error occurs: either no weights or something else.
    """
    try:
        is_pnet2 = model_builder.model_name[:6] == "P-NET2"

        if isNoneOrBlank(load_model_name):
            return None, None
        elif load_model_name[-1] == "*":
            #weights_path = os.path.join(f"{load_model_name.rstrip('*')}", f"{preproc_model_builder.model_name}_{model_builder.model_name}.h5")
            weights_path = load_model_name.rstrip('*')
            list_weights = list(Path(weights_path).glob("*.h5"))
            if len(list_weights) > 0:
                if f"{preproc_model_builder.model_name}_{model_builder.model_name}.h5" in list_weights:
                    weights_path = f"{preproc_model_builder.model_name}_{model_builder.model_name}.h5"
                else:
                    weights_path = list_weights[-1]
            else:
                return None, None
        else:
            weights_path = load_model_name
        
        if not os.path.exists(weights_path):
            return None, None
        
        print(f"\tWeights: {weights_path}")

        # load custom objects in tensorflow
        custom_objects = {}
        custom_objects['UnstackedRnn'] = keras_layers.UnstackedRnn
        custom_objects['ConcatRnn'] = keras_layers.ConcatRnn
        custom_objects['RandomShift'] = keras_layers.RandomShift
        custom_objects['RandomPermutation'] = keras_layers.RandomPermutation
        custom_objects['RandomCloudPermutation'] = keras_layers.RandomCloudPermutation
        custom_objects['CenterSequence'] = keras_layers.CenterSequence
        custom_objects['RandomRotation'] = keras_layers.RandomRotation
        if is_pnet2:
            custom_objects["Pointnet2Encoder"] = pointnet2.Pointnet2Encoder
            custom_objects["Pointnet2Decoder"] = pointnet2.Pointnet2Decoder
            custom_objects["IdentityLayer"] = keras_layers.IdentityLayer
            custom_objects["CnnSequence"] = keras_layers.CnnSequence
            custom_objects["Pointnet2DecoderCore"] = pointnet2.Pointnet2DecoderCore
            custom_objects['DecoderExtension'] = pointnet2.DecoderExtension
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            loaded_model = tf.keras.models.load_model(weights_path, compile=False)
        
        # net with multiple outputs is a mess to reconstruct, so ignore the partial models for that one
        if "split" in model_builder.decoder_type:
            return loaded_model, sub_models
        
        # split full model in submodels
        s_models = [] # list that will contain the sub models
        # use new preproc model rather than the saved one
        preproc_layers = sub_models[0]
        # append preprocessing model (note that the main model is always split in preprocessing and learning model)
        s_models.append(tf.keras.models.Model(inputs=preproc_layers.input, outputs=preproc_layers.output, name=sub_models[0].name))
        learn_layers = loaded_model.layers[1]
        # dinamically build the learning sub modules, should be compatible with any number and shape of models
        # as long as the output of a layer is the input of the next one (i.e. no skip connections or changes to input)
        input_layer = learn_layers.input
        out_layer = input_layer
        partial_outputs = []
        partial_outputs.append(out_layer)
        if is_pnet2:
            # manual build as we use skip connections and partial inputs
            states, enc_layer = learn_layers.layers[1](input_layer)
            gru_layer = learn_layers.layers[2](enc_layer)
            if (model_builder.rnn_type == 'identity_last_frame' or
                    ('return_sequences' in model_builder.rnn_args.keys() and 
                    not model_builder.rnn_args['return_sequences'])):
                states = [states[-1]] # take last time step and reinsert in a list
            dec_layer = learn_layers.layers[3]([gru_layer, states])
            # use states as output of encoder, it will also include the enc_layer
            # manual model recreation
            enc_model = tf.keras.models.Model(inputs=input_layer, outputs=states, name="Pointnet2Encoder")
            rnn_model = tf.keras.models.Model(inputs=input_layer, outputs=gru_layer, name="Pointnet2Rnn")
            dec_model = tf.keras.models.Model(inputs=input_layer, outputs=dec_layer, name="Pointnet2Decoder")
            s_models.extend([enc_model,rnn_model,dec_model])
        else:
            for l in learn_layers.layers[1:]: # skip input 
                out_layer = l(out_layer)
                partial_outputs.append(out_layer)
            for m in sub_models[1:]: # skip preproc
                t_model = tf.keras.models.Model(inputs=input_layer, outputs=partial_outputs[len(m.layers)-1], name=m.name)
                s_models.append(t_model)

        loaded_model.compile(loss = full_model.loss, optimizer=full_model.optimizer, metrics=full_model.metrics)
        return loaded_model, s_models
    except (FileNotFoundError, OSError) as e:
        return None, None


def load_checkpoint(model, in_model_dir, mode="best"):
    """
    Simply load the weights from a checkpoint.
    It is assumed that the model is compatible with the weights.
    mode can be either 'best' or 'last'.
    Best is used for the weights associated to the best validation error,
    last to the weights of the last training epoch.
    Use save_checkpoint function to save the last checkpoints and the get_checkpoints_callback to save the best.
    """
    try:
        if mode == "best":
            checkpoint_dir = "checkpoints"
        elif mode == "last":
            checkpoint_dir = "checkpoints_last"
        checkpoint_path = os.path.join(in_model_dir, checkpoint_dir, "ckpt.hdf5")
        model.load_weights(checkpoint_path)
        return model
    except Exception: # no folder found
        return None


def save_checkpoint(model, save_model_dir):
    """
    Save the last weights of the training.
    """
    checkpoint_path = os.path.join(save_model_dir, "checkpoints_last")
    Path(checkpoint_path).mkdir(exist_ok=True, parents=True)
    checkpoint_path = os.path.join(checkpoint_path, "ckpt.hdf5")
    model.save_weights(checkpoint_path)


def prepare_dataset_for_training(split_train_ds, split_val_ds, preproc_model_builder, model_builder, train_batch_size, target_train=None, target_val=None, **dataset_vars):
    """
    Prepare the dataset for training (cache, prefetch and data mapping).
    split_train_ds and split_val_ds are the training and val datasets. Ignore the fact that they start with split, 
    it's from an older version of the code.

    Shift and rotation are not applied to the validation set as we don't use it for training and we want a real validation result.

    target_train and target_val are the markers if available
    """
    # Note: map function keeps the order of the data
    my_train_ds = preproc_model_builder.apply_data_preproc(split_train_ds)
    if target_train is None:
        my_train_ds = tf.data.Dataset.zip((my_train_ds, my_train_ds))
    else:
        my_train_ds = tf.data.Dataset.zip((my_train_ds, target_train))

    if split_val_ds is not None:
        my_val_ds = preproc_model_builder.apply_data_preproc(split_val_ds)
        if target_val is None:
            my_val_ds = tf.data.Dataset.zip((my_val_ds, my_val_ds))
        else:
            my_val_ds = tf.data.Dataset.zip((my_val_ds, target_val))
    else:
        my_val_ds = split_val_ds
    
    if dataset_vars['dataset_shift_apply']:
        my_train_ds = apply_dataset_shift(my_train_ds, **dataset_vars)
        #my_val_ds = apply_dataset_shift(my_val_ds, **dataset_vars) 
    
    if dataset_vars['dataset_normalize_after_center_seq']:
        if dataset_vars['dataset_center_seq']:
            my_train_ds = apply_center_sequence(my_train_ds, **dataset_vars)
            my_val_ds = apply_center_sequence(my_val_ds, **dataset_vars)
        
        if dataset_vars['dataset_normalization_apply']:
            my_train_ds = apply_dataset_normalization(my_train_ds, **dataset_vars)
            my_val_ds = apply_dataset_normalization(my_val_ds, **dataset_vars)
    else:
        if dataset_vars['dataset_normalization_apply']:
            my_train_ds = apply_dataset_normalization(my_train_ds, **dataset_vars)
            my_val_ds = apply_dataset_normalization(my_val_ds, **dataset_vars)

        if dataset_vars['dataset_center_seq']:
            my_train_ds = apply_center_sequence(my_train_ds, **dataset_vars)
            my_val_ds = apply_center_sequence(my_val_ds, **dataset_vars)
    
    if dataset_vars['dataset_rotation_apply']:
        my_train_ds = apply_dataset_rotation(my_train_ds, **dataset_vars)
        #my_val_ds = apply_dataset_rotation(my_val_ds, **dataset_vars)
    
    if dataset_vars['dataset_shuffle']:
        my_train_ds = my_train_ds.shuffle()
        #my_val_ds = my_val_ds.shuffle()
    
    if "split" in model_builder.decoder_type:
        # multi output model
        if "model_dec_unsupervised" in model_builder.decoder_args.keys() and model_builder.decoder_args['model_dec_unsupervised']['target'] == "x":
            my_train_ds = my_train_ds.map(lambda x,y: (x, {"MarkersDecoder": y, "CloudDecoder": x}))
            my_val_ds = my_val_ds.map(lambda x,y: (x, {"MarkersDecoder": y, "CloudDecoder": x}))
        else:
            # target is y
            my_train_ds = my_train_ds.map(lambda x,y: (x, {"MarkersDecoder": y, "CloudDecoder": y}))
            my_val_ds = my_val_ds.map(lambda x,y: (x, {"MarkersDecoder": y, "CloudDecoder": y}))
    
    my_train_ds = my_train_ds.batch(train_batch_size).cache().prefetch(tf.data.AUTOTUNE)
    if my_val_ds is not None:
        my_val_ds = my_val_ds.batch(1).cache().prefetch(tf.data.AUTOTUNE)
    return my_train_ds, my_val_ds


def train(model_builder, full_model, train_ds, val_ds, train_vars, model_dir, verbose=0):
    # train the model
    train_epochs = train_vars['train_epochs']
    my_callbacks = list(train_vars['callbacks'].keys())
    all_callbacks = []
    if "checkpoints" in my_callbacks:
        checkpoint_folder = os.path.join(model_dir, 'checkpoints/') # trailing slash is necessary
        Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
        all_callbacks.append(get_checkpoints_callback(out_folder=checkpoint_folder, monitor=train_vars['callbacks']['checkpoints']['monitor'], 
                mode=train_vars['callbacks']['checkpoints']['mode']))
    if "print_time" in my_callbacks:
        all_callbacks.append(get_print_time_callback(t_delta=train_vars['callbacks']['print_time']['timedelta']))
    if "lr" in my_callbacks:
        all_callbacks.append(get_lr_callback(epoch_divider=train_vars['callbacks']["lr"]["epoch_divider"]))
    if "es" in my_callbacks:
        all_callbacks.append(get_es_callback(patience=train_vars['callbacks']["es"]["patience"]))

    history = full_model.fit(train_ds, 
                        validation_data=val_ds,
                        epochs=train_epochs,
                        verbose=verbose,
                        callbacks=all_callbacks)
    model_builder.load_train_history(history)
