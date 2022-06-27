import os
from datetime import datetime
import json
import h5py # keep it


from splashlib.nn.utils_data import setup_dataset, convert_dataset, setup_markers
from splashlib.nn.k_utils import isNotBlank
from splashlib.nn.utils_model import model_from_builder
import splashlib.nn.utils_params as upars
from splashlib.nn.utils_flow import *
from splashlib.nn.utils import *

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# save model folder
save_model_dir = f"notebook_dumps/models_out/{current_time}"

def main():
    # parse input vars
    seed, dataset_vars, preproc_vars, model_vars, train_vars, save_model_name, load_model_name = upars.parse_args()
    current_path = os.path.dirname(__file__)
    setup_env(current_path=current_path, save_model_dir=save_model_dir, seed=seed, deterministic=False)
    # disable gpu if necessary
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
    my_train_ds, my_val_ds = prepare_dataset_for_training(split_train_ds=split_train_ds, 
                                    split_val_ds=split_val_ds, 
                                    preproc_model_builder=preproc_model_builder, 
                                    model_builder=model_builder,
                                    train_batch_size=train_batch_size, 
                                    target_train=target_train_ds, 
                                    target_val=target_val_ds, 
                                    **dataset_vars)
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
    print(f"\tStarting training procedure at: {datetime.now()}")
    train(model_builder, full_model, my_train_ds, my_val_ds, train_vars, save_model_dir)
    # save trained model
    print("------------------------------")
    print("\tSaving checkpoint")
    save_checkpoint(full_model, save_model_dir)
    print()
    print("\tSaving complete model")
    full_model.save(os.path.join(save_model_dir, f'{save_name}.h5'))
    print()
    print("------------------------------")
    print("\tSaving loss")
    save_loss(model_builder, save_model_dir)
    print()
    print("------------------------------")
    print(f"Completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
