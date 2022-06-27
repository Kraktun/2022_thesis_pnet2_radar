
"""
Class that represents the learning model, both the structure and the training parameters.
"""

class ModelBuilderK:
    
    def __init__(self, model_name, model_type,
                    encoder_type, rnn_type, decoder_type,
                    encoder_args, rnn_args, decoder_args,
                    loss=None, optimizer=None, metrics=None):
        # main vars
        self.model_name = model_name
        self.model_type = model_type
        # model structure
        self.encoder_type = encoder_type
        self.rnn_type = rnn_type
        self.decoder_type = decoder_type
        self.encoder_args = encoder_args
        self.rnn_args = rnn_args
        self.decoder_args = decoder_args
        # learning
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.history = {'loss': [], 'val_loss': [], 'epochs': [], "cloud_loss": [], 
            "cloud_val_loss": [], "marker_loss": [], "marker_val_loss": []}

    def in_shape(self):
        return self.time_steps, self.max_points, self.n_feats
    
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "encoder_type": self.encoder_type,
            "rnn_type": self.rnn_type,
            "decoder_type": self.decoder_type,
            "encoder_args": self.encoder_args,
            "rnn_args": self.rnn_args,
            "decoder_args": self.decoder_args,
            "loss": self.loss,
            "optimizer": self.optimizer,
            "metrics": self.metrics,
        }
    
    def print_dict(self):
        print("{")
        for k,v in self.to_dict().items():
            print(f"\t{k}: {v}")
        print("}")

    def load_train_history(self, history):
        self.history['epochs'].append(history.epoch)
        self.history['loss'].append(history.history['loss'])
        self.history['val_loss'].append(history.history['val_loss'])
        if 'MarkersDecoder_loss' in history.history.keys():
            self.history['marker_loss'].append(history.history['MarkersDecoder_loss'])
            self.history['cloud_loss'].append(history.history['CloudDecoder_loss'])
            self.history['marker_val_loss'].append(history.history['val_MarkersDecoder_loss'])
            self.history['cloud_val_loss'].append(history.history['val_CloudDecoder_loss'])
