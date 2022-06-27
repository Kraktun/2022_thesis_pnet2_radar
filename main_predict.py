import os
from datetime import datetime
import json
import h5py # keep it
from collections.abc import Mapping
import numpy as np


from splashlib.nn.utils_data import setup_dataset, convert_dataset, setup_markers
from splashlib.nn.k_utils import isNotBlank
from splashlib.nn.utils_model import model_from_builder
import splashlib.nn.utils_params as upars
from splashlib.nn.utils_flow import *
from splashlib.nn.utils import *

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# save model folder
save_model_dir = f"notebook_dumps/models_out/{current_time}_predict"

def main():
    seed, dataset_vars, preproc_vars, model_vars, train_vars, save_model_name, load_model_name = upars.parse_args()
    current_path = os.path.dirname(__file__)
    setup_env(current_path=current_path, save_model_dir=save_model_dir, seed=seed, deterministic=False)
    # disable gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # dataset
    print("------------------------------")
    print("\tPreparing dataset")
    global_feats = dataset_vars.pop('global_feats', None)
    split_train_ds, split_val_ds = setup_dataset(**dataset_vars)
    
    # preprocessing model
    print("------------------------------")
    print("\tPreparing preprocessing model")
    preproc_model_builder = get_preprocessing_model(time_steps=dataset_vars['time_steps'], max_points=dataset_vars['max_points'], global_feats=global_feats, **preproc_vars)
    with open(os.path.join(save_model_dir, f"{preproc_model_builder.model_name}.preproc.params.json"), "w") as f:
        json.dump(preproc_vars, f, indent=4)
    
    # learning model
    print("------------------------------")
    print("\tPreparing main model")
    model_builder = get_model(**model_vars)
    with open(os.path.join(save_model_dir, f"{model_builder.model_name}.model.params.json"), "w") as f:
        json.dump(model_vars, f, indent=4)
    
    # preprocess dataset
    print("------------------------------")
    print("\tProcessing dataset")
    train_batch_size = train_vars['train_batch_size']
    if model_builder.model_type == "supervised":
        target_train_ds, target_val_ds = setup_markers(**dataset_vars)
    else:
        target_train_ds = None
        target_val_ds = None

    # Disable dataset normalization
    was_normalized = dataset_vars['dataset_normalization_apply']
    dataset_vars['dataset_normalization_apply'] = False

    my_train_ds, my_val_ds = prepare_dataset_for_training(split_train_ds=split_train_ds, 
                                    split_val_ds=split_val_ds, 
                                    preproc_model_builder=preproc_model_builder, 
                                    model_builder=model_builder,
                                    train_batch_size=train_batch_size, 
                                    target_train=target_train_ds, 
                                    target_val=target_val_ds, 
                                    **dataset_vars)

    # Compute the normalization on the ds
    y_source = []
    x_source = []
    if was_normalized:
        norm_vals = []
        for x,y in my_val_ds.unbatch():
            if isinstance(y, Mapping):
                y = y['MarkersDecoder']
            x_source.append(x[-1]) # last frame
            y_source.append(y)
            x_min, x_interval, y_min, y_interval = apply_normalization(x, y, **dataset_vars)
            norm_vals.append((x_min, x_interval, y_min, y_interval))
        print(f"mean NORM_MIN: {tf.reduce_mean(y_min, axis=0)}")
        print(f"mean NORM_INTERVAL: {tf.reduce_mean(y_interval, axis=0)}")
        # recompute the correct datasets
        dataset_vars['dataset_normalization_apply'] = True
        my_train_ds, my_val_ds = prepare_dataset_for_training(split_train_ds=split_train_ds, 
                                    split_val_ds=split_val_ds, 
                                    preproc_model_builder=preproc_model_builder, 
                                    model_builder=model_builder,
                                    train_batch_size=train_batch_size, 
                                    target_train=target_train_ds, 
                                    target_val=target_val_ds, 
                                    **dataset_vars)
    else:
        for x,y in my_val_ds.unbatch():
            if isinstance(y, Mapping):
                y = y['MarkersDecoder']
            y_source.append(y)
            x_source.append(x[-1]) # last frame
    
    print(f"Train ds size: {len(my_train_ds)*train_batch_size}")
    if my_val_ds is not None:
        print(f"Val ds size: {len(my_val_ds)}")
    
    # assemble and compile preproc+learning model
    print("------------------------------")
    print("\tAssembling models")
    full_model, sub_models = model_from_builder(preproc_model_builder, model_builder)
    if "load_from_checkpoint" in train_vars.keys() and train_vars['load_from_checkpoint']:
        checkpoint_mode = "best"
        if "checkpoint_mode" in train_vars.keys():
            checkpoint_mode = train_vars['checkpoint_mode']
        in_model_dir = load_model_name.rstrip("*")
        in_model_dir = Path(in_model_dir)
        if in_model_dir.is_file():
            in_model_dir = in_model_dir.parent()
        loaded_model = load_checkpoint(full_model, in_model_dir, checkpoint_mode)
    else:
        loaded_model, sub_models = load_saved(preproc_model_builder=preproc_model_builder, model_builder=model_builder, 
                                    full_model=full_model, sub_models=sub_models, load_model_name=load_model_name)
    if loaded_model:
        print("\t\tUsing pre-trained weights")
        full_model = loaded_model
    else:
        print("\t\tTraining from scratch")

    save_name = f"{preproc_model_builder.model_name}_{model_builder.model_name}"
    if isNotBlank(save_model_name):
        save_name = save_model_name
    print(f"\n\tMODEL NAME: {save_name}\n")
    
    print("------------------------------")
    print("\tPrinting models\n")
    if "split" in model_builder.decoder_type:
        with open(os.path.join(save_model_dir, f"{save_name}.summary.txt"), "a") as f:
            full_model.summary(line_length=100, print_fn=lambda x: f.write(x + '\n'))
    else:
        for l in full_model.layers:
            with open(os.path.join(save_model_dir, f"{save_name}.summary.txt"), "a") as f:
                l.summary(line_length=100, print_fn=lambda x: f.write(x + '\n'))
    
    # save dataset params
    with open(os.path.join(save_model_dir, f"{save_name}.dataset.params.json"), "w") as f:
        json.dump(dataset_vars, f, indent=4)
    with open(os.path.join(save_model_dir, f"{save_name}.train.params.json"), "w") as f:
        json.dump(train_vars, f, indent=4)
    # train
    print("------------------------------")
    print(f"\tStarting predicting procedure at: {datetime.now()}")
    #train(model_builder, full_model, my_train_ds, my_val_ds, train_vars, save_model_dir)
    # predict the labels
    y_hat = full_model.predict(my_val_ds, verbose=1)
    if isinstance(y_hat, list):
        print("------------------------------")
        print("\tMulti-output model")
        y_hat_cloud = y_hat[1]
        y_hat = y_hat[0]
    else:
        y_hat_cloud = None
    save_stack = []
    ppd_stack = []
    masks = []
    diff_per_markers = []
    denormed_y_hat = []
    for j in range(y_source[0].shape[-2]):
        diff_per_markers.append([])
    for i, y_hat_i in enumerate(y_hat):
        y_hat_i = tf.squeeze(y_hat_i)
        if was_normalized:
            # convert the normalized y to the un-normalized state
            y_rev = []
            y_ax_min = norm_vals[i][2]
            y_ax_min = tf.squeeze(y_ax_min)
            y_ax_interval = norm_vals[i][3]
            y_ax_interval = tf.squeeze(y_ax_interval)
            for ax in range(y_hat_i.shape[-1]):
                y_new = y_hat_i[...,ax]
                if ax in dataset_vars['dataset_normalization_axis'] and ax < 3: # only normalize xyz, not the column to enable/disable markers
                    t_y_min = y_ax_min[...,ax]
                    t_y_int = y_ax_interval[...,ax]
                    y_new = y_new*t_y_int+t_y_min
                y_rev.append(y_new)
            y_hat_i = tf.stack(y_rev, axis=-1)
        denormed_y_hat.append(y_hat_i)
        # remove nan values
        mask = tf.where(tf.squeeze(y_source[i][...,-1]) > 0)
        mask = tf.reshape(mask, [-1])
        masks.append(mask)
        y_source_t = tf.gather(tf.squeeze(y_source[i]), mask)
        y_hat_i = tf.gather(tf.squeeze(y_hat_i), mask)
        diff_x = y_hat_i[...,0] - y_source_t[...,0] 
        diff_y = y_hat_i[...,1] - y_source_t[...,1]
        diff_z = y_hat_i[...,2] - y_source_t[...,2]
        np_arr = np.stack([diff_x, diff_y, diff_z]).T
        save_stack.append(np_arr)
        ppd = array_ppd(diff_x, diff_y, diff_z)
        ppd_stack.append(ppd)
        ind_rev = 0
        for m in range(len(diff_per_markers)):
            if m in mask:
                diff_per_markers[m].append([diff_x[ind_rev], diff_y[ind_rev], diff_z[ind_rev]])
                ind_rev = ind_rev+1
    #save_stack = np.array(save_stack)
    stack_folder = Path(os.path.join(save_model_dir, "stack_loss"))
    stack_folder.mkdir(exist_ok=True, parents=True)
    for i in range(len(save_stack)):
        stack_file = stack_folder / f"stack_loss_{i}.csv"
        np.savetxt(stack_file, np.array(save_stack[i]), delimiter=",", fmt='%1.5f')
        mask_file = stack_folder / f"stack_loss_{i}_mask.csv"
        np.savetxt(mask_file, np.array(masks[i]), delimiter=",", fmt='%1.5f')

    # diff_per_markers is a list where for marker i diff_per_markers[i] = list of lists [x,y,z], 
    # note that len(diff_per_markers[i]) may be different from len(diff_per_markers[j])
    # if some marker never appears, just remove them and call it a day (print a warning first)
    for i,m in enumerate(diff_per_markers):
        if len(m) == 0:
            print("--------------------")
            print(f"\tMarker at position {i} is always NaN in the dataset")
    diff_per_markers = [m for m in diff_per_markers if len(m)>0]
    mean_per_marker = [np.mean(np.abs(tup), axis=0) for tup in diff_per_markers]
    markers_str = '\n'.join([str(m) for m in mean_per_marker])
    print("------------------------------")
    print(f"Mean per marker: \n{markers_str}")
    mean_file = stack_folder / f"mean_loss_per_marker.csv"
    np.savetxt(mean_file, mean_per_marker, delimiter=",", fmt='%1.5f')
	
    std_per_marker = [np.std(tup, axis=0) for tup in diff_per_markers]
    std_str = '\n'.join([str(m) for m in std_per_marker])
    print("------------------------------")
    print(f"Std per marker: \n{std_str}")
    std_file = stack_folder / f"std_per_marker.csv"
    np.savetxt(std_file, std_per_marker, delimiter=",", fmt='%1.5f')

    mean_per_axis = np.mean(mean_per_marker, axis=0).T
    print("------------------------------")
    print(f"Mean per axis: {mean_per_axis}")
    mean_file2 = stack_folder / f"mean_loss_xyz.csv"
    np.savetxt(mean_file2, mean_per_axis, delimiter=",", fmt='%1.5f')

    # linear distance from mean per axis
    lin_dist = np.sqrt(np.sum(mean_per_axis**2))
    print("------------------------------")
    print(f"Linear distance from mean: {lin_dist}")
    lin_dist_file = stack_folder / f"lin_dist_xyz.csv"
    np.savetxt(lin_dist_file, [lin_dist], delimiter=",", fmt='%1.5f')

    # linear distance for each marker, then mean
    lin_dist_2 = np.sqrt(np.sum(np.array(mean_per_marker)**2, axis=1)).T
    print("------------------------------")
    print(f"Linear distance per markers: {lin_dist_2}")
    lin_dist_file_2 = stack_folder / f"lin_dist_markers.csv"
    np.savetxt(lin_dist_file_2, lin_dist_2, delimiter=",", fmt='%1.5f')

    lin_dist_3 = np.mean(lin_dist_2)
    print("------------------------------")
    print(f"Mean linear distance from markers: {lin_dist_3}")
    lin_dist_file_3 = stack_folder / f"lin_dist_markers_mean.csv"
    np.savetxt(lin_dist_file_3, [lin_dist_3], delimiter=",", fmt='%1.5f')

    full_mean = np.mean(mean_per_marker)
    print("------------------------------")
    print(f"Full mean: {full_mean}")

    # the selective per point squared distance to check against val loss from training
    ppd_file = stack_folder / f"mean_loss_ppd.csv"
    mean_ppd = np.mean(ppd_stack)
    np.savetxt(ppd_file, [mean_ppd], delimiter=",", fmt='%1.5f')
    print(f"PPD: {mean_ppd}")

    # save predictions and source files as a global list
    out_folder = Path(os.path.join(save_model_dir, "results"))
    source_dir = out_folder / "source"
    predict_dir = out_folder / "predict"
    source_dir.mkdir(exist_ok=True, parents=True)
    predict_dir.mkdir(exist_ok=True, parents=True)
    for i, (y_s, y_p) in enumerate(zip(y_source, denormed_y_hat)):
        y_s = tf.squeeze(y_s)
        y_p = tf.squeeze(y_p)
        out_source = source_dir / f"{i:03d}.csv"
        out_pred = predict_dir / f"{i:03d}.csv"
        np.savetxt(out_source, y_s, delimiter=",", fmt='%1.5f')
        np.savetxt(out_pred, y_p, delimiter=",", fmt='%1.5f')
    
    # print also clouds (note: not un-normalized)
    source_cloud_dir = out_folder / "source_cloud"
    source_cloud_dir.mkdir(exist_ok=True, parents=True)
    for i, x_sc in enumerate(x_source):
        x_sc = tf.squeeze(x_sc)
        out_source_cloud = source_cloud_dir / f"{i:03d}.csv"
        np.savetxt(out_source_cloud, x_sc, delimiter=",", fmt='%1.5f')
    if y_hat_cloud is not None:
        pred_cloud_dir = out_folder / "pred_cloud"
        pred_cloud_dir.mkdir(exist_ok=True, parents=True)
        for i, y_pc in enumerate(y_hat_cloud):
            y_pc = tf.squeeze(y_pc)
            out_pred_cloud = pred_cloud_dir / f"{i:03d}.csv"
            np.savetxt(out_pred_cloud, y_pc, delimiter=",", fmt='%1.5f')

    print()
    print("------------------------------")
    print(f"Completed at: {datetime.now()}")

def apply_normalization(x, y, dataset_normalization_xyz, dataset_normalization_sequence, dataset_normalization_axis, **dataset_vars):
    # same function from utils_data, reduced for only x
    # use normalization for x also for y
    if isinstance(dataset_normalization_sequence, bool):
        dataset_normalization_sequence = [dataset_normalization_sequence]*len(dataset_normalization_axis)
    x_shape = x.shape

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
    if y.shape[0] < x.shape[0]:
        y_ax_min = y_ax_min[-y.shape[0]]
        y_ax_max = y_ax_max[-y.shape[0]]
        if len(y_ax_min.shape) < len(y.shape):
            y_ax_min = tf.expand_dims(y_ax_min, 0)
            y_ax_max = tf.expand_dims(y_ax_max, 0)
    y_ax_interval = y_ax_max - y_ax_min
    # replace 0 values with 1 to avoid NaN in the division
    y_ax_interval = tf.where(y_ax_interval==0.0, 1.0, y_ax_interval)
    return x_ax_min, x_ax_interval, y_ax_min, y_ax_interval

def array_ppd(diff_x, diff_y, diff_z):
    return np.mean(diff_x**2 + diff_y**2 + diff_z**2)

if __name__ == "__main__":
    main()
