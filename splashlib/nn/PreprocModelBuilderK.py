import tensorflow as tf

"""
Class that represents the preprocessing associated to a model architecture.
Allows to reuse the same architecture with different types of preprocessing.
"""

class PreprocModelBuilderK:
    
    def __init__(self, model_name, model_type,
                points_in_frame, num_features,
                in_time_steps, # necessary value also if no split layer is needed
                out_time_steps=None, # if out_time_steps is specified, a split layer will be added 
                out_time_steps_stride=None, 
                standardize=False, skip_standardization_axis=[], eps=1e-6,# note that skip_standardization_axis is applied after feature reduction, so it's relative to keep_features
                keep_features=[], # remove all columns not in this list from the last axis
                add_channel=False, # add one extra axis at the end (i.e. channel axis)
                random_shift=False, random_shift_prob=0.3, random_shift_interval=(-1,1), # control random shift augmentation
                random_permutation=False, random_permutation_prob=0.3, random_permutation_axis=[1,2,3], # control random permutation on x,y,z axis
                random_shuffle=False, random_shuffle_prob=0.3, # control random shuffle of points in a frame
                random_rotation=False, random_rotation_prob=0.3, random_rotation_angles=[[-1,1]]*3 # control random rotation
    ):
        # main vars
        self.model_name = model_name
        self.model_type = model_type
        # structure
        self.points_in_frame = points_in_frame
        self.num_features = num_features
        self.in_time_steps = in_time_steps
        self.out_time_steps = out_time_steps
        if self.out_time_steps is None or self.out_time_steps < 1:
            self.out_time_steps = self.in_time_steps
            self.out_time_steps_stride = None
        else:
            self.out_time_steps_stride = out_time_steps_stride
        self.standardize = standardize
        self.skip_standardization_axis = skip_standardization_axis
        self.keep_features = keep_features
        self.add_channel = add_channel
        self.eps = eps
        self.random_shift = random_shift
        self.random_shift_prob = random_shift_prob
        self.random_shift_interval = random_shift_interval
        self.random_permutation = random_permutation
        self.random_permutation_prob = random_permutation_prob
        self.random_permutation_axis = random_permutation_axis
        self.random_shuffle = random_shuffle
        self.random_shuffle_prob = random_shuffle_prob
        self.random_rotation = random_rotation
        self.random_rotation_prob = random_rotation_prob
        self.random_rotation_angles = random_rotation_angles
    
    def apply_data_preproc(self, dataset):
        if self.has_reduce_features():
            dataset = dataset.map(lambda x: tf.gather(x, self.keep_features, axis=-1))
        if self.standardize:
            dataset = dataset.map(lambda x: self._standardize_tensor(x))
        return dataset
    
    def _standardize_tensor(self, inputs):
        # NOTE: MAY BE CURRENTLY BROKEN (the layer is ok, this function maybe not)
        feat_axis = 3
        for i in range(self.out_features()):
            if i in self.skip_standardization_axis:
                p = inputs[:,:,i]
                p = tf.expand_dims(p, -1)
                if i == 0:
                    t = p
                else:
                    t = tf.concat([t, p], axis=feat_axis)
            else:
                p = (inputs[:,:,i] - tf.math.reduce_mean(inputs[:,:,i], axis=feat_axis-1, keepdims=True)) / (tf.math.reduce_std(inputs[:,:,i], axis=feat_axis-1, keepdims=True) + self.eps)
                p = tf.expand_dims(p, -1)
                if i == 0:
                    t = p
                else:
                    t = tf.concat([t, p], axis=feat_axis)
        return t

    def in_shape(self):
        if self.has_split_layer():
            return self.points_in_frame, self.out_features()
        return self.in_time_steps, self.points_in_frame, self.out_features()
    
    def out_shape(self):
        out_feats = self.out_features()
        if self.add_channel:
            return self.out_time_steps, self.points_in_frame, out_feats, 1
        return self.out_time_steps, self.points_in_frame, out_feats
    
    def has_split_layer(self):
        # return true if preprocessing model will include a split layer
        return self.out_time_steps_stride is not None and self.out_time_steps_stride > 0
    
    def has_reduce_features(self):
        return len(self.keep_features) > 0
    
    def out_features(self):
        if self.has_reduce_features():
            return len(self.keep_features)
        return self.num_features
    
    def to_dict(self):
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "points_in_frame": self.points_in_frame,
            "num_features": self.num_features,
            "in_time_steps": self.in_time_steps,
            "out_time_steps" : self.out_time_steps,
            "time_steps_stride": self.out_time_steps_stride,
            "standardize": self.standardize,
            "skip_standardization_axis": self.skip_standardization_axis,
            "eps": self.eps,
            "keep_features": self.keep_features,
            "add_channel": self.add_channel,
            "random_shift": self.random_shift,
            "random_shift_prob": self.random_shift_prob,
            "random_shift_interval": self.random_shift_interval,
            "random_permutation": self.random_permutation,
            "random_permutation_prob": self.random_permutation_prob,
            "random_permutation_axis": self.random_permutation_axis,
            "random_shuffle": self.random_shuffle,
            "random_shuffle_prob": self.random_shuffle_prob,
            "random_rotation": self.random_rotation,
            "random_rotation_prob": self.random_rotation_prob,
            "random_rotation_angles": self.random_rotation_angles,
        }
    
    def print_dict(self):
        print("{")
        for k,v in self.to_dict().items():
            print(f"\t{k}: {v}")
        print("}")
