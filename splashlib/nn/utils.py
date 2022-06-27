import tensorflow as tf
import numpy as np
from pathlib import Path
import os

def ConfigGPU():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)
    else:
        print('No GPU available.')

def normalize(data):
    maxs = np.max(data, axis=0)
    mins = np.min(data, axis=0)
    if (maxs - mins) != 0:
        norm_data = (data - mins)/(maxs - mins)
    else:
        norm_data = data
    return norm_data

def isNaN(num):
    return num != num

def is_pos_def(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M + M.transpose()) > 1e-10)

def save_loss(model_builder, save_model_dir):
    """
    Save loss as csv files.
    """
    train_loss = np.array(model_builder.history['loss']).T
    epochs = np.array(model_builder.history['epochs']).T
    val_loss = np.array(model_builder.history['val_loss']).T
    loss_folder = Path(os.path.join(save_model_dir, "loss"))
    loss_folder.mkdir(exist_ok=True, parents=True)
    train_file = loss_folder / "train_loss.csv"
    epochs_file = loss_folder / "epochs.csv"
    val_file = loss_folder / "val_loss.csv"
    np.savetxt(train_file, train_loss, delimiter=",", fmt='%1.5f')
    np.savetxt(epochs_file, epochs, delimiter=",", fmt='%1.5f')
    np.savetxt(val_file, val_loss, delimiter=",", fmt='%1.5f')
    if len(model_builder.history['cloud_loss']) > 0:
        cloud_loss = np.array(model_builder.history['cloud_loss']).T
        marker_loss = np.array(model_builder.history['marker_loss']).T
        cloud_val_loss = np.array(model_builder.history['cloud_val_loss']).T
        marker_val_loss = np.array(model_builder.history['marker_val_loss']).T
        cloud_file = loss_folder / "cloud_loss.csv"
        cloud_val_file = loss_folder / "cloud_val_loss.csv"
        marker_file = loss_folder / "marker_loss.csv"
        marker_val_file = loss_folder / "marker_val_loss.csv"
        np.savetxt(cloud_file, cloud_loss, delimiter=",", fmt='%1.5f')
        np.savetxt(cloud_val_file, cloud_val_loss, delimiter=",", fmt='%1.5f')
        np.savetxt(marker_file, marker_loss, delimiter=",", fmt='%1.5f')
        np.savetxt(marker_val_file, marker_val_loss, delimiter=",", fmt='%1.5f')
            
