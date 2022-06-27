import tensorflow as tf
from datetime import datetime, timedelta
import os

"""
This file includes some functions to generate callbacks to be used with tensorflow.
They are self-explanatory I believe.
get_print_time_callback is particularly useful if you plan to use the cluster at dei 
to check the training (disable verbose during training in this case).
"""

def get_checkpoints_callback(out_folder, monitor='val_loss', mode='min'):
    checkpoint_path = os.path.join(out_folder, "ckpt.hdf5")
    return tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor=monitor,
        save_best_only=True,
        save_weights_only=True,
        mode=mode,
        save_freq="epoch"
    )


def get_es_callback(patience):
    return tf.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=patience)


def get_lr_callback(epoch_divider=50, verbose=0):

    def lr_scheduler(epoch, lr):
        # divide lr by 10 after epoch_divider epochs
        index_epoch = epoch % epoch_divider
        if index_epoch == 0 and epoch > 0:
            lr = lr/10
        return lr
    
    return tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=verbose)


def get_print_time_callback(t_delta):
    # t_delta is the time delta to add if for instance you are running it on a server with a different time 

    class PrintTimesCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            print(f"Start training at: {datetime.now() + timedelta(hours=t_delta)}")

        def on_train_end(self, logs=None):
            print(f"End training at: {datetime.now() + timedelta(hours=t_delta)}")

        def on_epoch_begin(self, epoch, logs=None):
            print(f"Start epoch {epoch} at time: {datetime.now() + timedelta(hours=t_delta)}")

        def on_epoch_end(self, epoch, logs=None):
            try:
                loss = logs['loss']
            except Exception:
                loss = 'unknown'
            try:
                val_loss = logs['val_loss']
            except Exception:
                val_loss = 'unknown'
            print(f"End epoch {epoch} at time: {datetime.now() + timedelta(hours=t_delta)}\n\tCurrent loss: {loss}, val loss: {val_loss}")
            
            if 'MarkersDecoder_loss' in logs.keys():
                marker_loss = logs['MarkersDecoder_loss']
                cloud_loss = logs['CloudDecoder_loss']
                marker_val_loss = logs['val_MarkersDecoder_loss']
                cloud_val_loss = logs['val_CloudDecoder_loss']
                print(f"\tCurrent cloud loss: {cloud_loss}, cloud val loss: {cloud_val_loss}")
                print(f"\tCurrent marker loss: {marker_loss}, marker val loss: {marker_val_loss}")

    return PrintTimesCallback()
