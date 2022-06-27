"""
Some basic examples of model parameters.
"""

pointnet1_Conv3D = {
    "stride"           : [1,1,1],
    "filters"          : [64,64,64,128,1024],
    "padding"          : "VALID",
    "data_format"      : "channels_last",
    "use_xavier"       : True,
    "stddev"           : 1e-3,
    "weight_decay"     : None,
    "activation_fn"    : "relu",
    "batch_norm"       : True,
    "batch_norm_decay" : 0.9
}

pointnet1_Conv3D_short = {
    "stride"           : [1,1,1],
    "filters"          : [32,32,32,64,64],
    "padding"          : "VALID",
    "data_format"      : "channels_last",
    "use_xavier"       : True,
    "stddev"           : 1e-3,
    "weight_decay"     : None,
    "activation_fn"    : "relu",
    "batch_norm"       : True,
    "batch_norm_decay" : 0.9
}

pointnet1_Conv2D = {
    "stride"           : [1,1],
    "filters"          : [64,64,64,128,1024],
    "padding"          : "VALID",
    "data_format"      : "channels_last",
    "use_xavier"       : True,
    "stddev"           : 1e-3,
    "weight_decay"     : None,
    "activation_fn"    : "relu",
    "batch_norm"       : True,
    "batch_norm_decay" : 0.9
}

pointnet2_sem_seg_enc = {
    'sa_settings': [
        {'npoint': 1024, 'radius': 0.1, 'nsample': 32, 'filters': [32,32,64], 'group_all': False, 'name': 'sa_module_1'},
        {'npoint': 256, 'radius': 0.2, 'nsample': 32, 'filters': [64,64,128], 'group_all': False, 'name': 'sa_module_2'},
        {'npoint': 64, 'radius': 0.4, 'nsample': 32, 'filters': [128,128,256], 'group_all': False, 'name': 'sa_module_3'},
        {'npoint': 16, 'radius': 0.8, 'nsample': 32, 'filters': [256,256,512], 'group_all': False, 'name': 'sa_module_4'}    
    ]
}

pointnet2_sem_seg_enc_reduced = {
    'sa_settings': [
        {'npoint': 64, 'radius': 0.1, 'nsample': 16, 'filters': [8,8,16], 'group_all': False, 'name': 'sa_module_1'},
        {'npoint': 32, 'radius': 0.2, 'nsample': 16, 'filters': [16,16,32], 'group_all': False, 'name': 'sa_module_2'},
        {'npoint': 16, 'radius': 0.4, 'nsample': 16, 'filters': [32,32,64], 'group_all': False, 'name': 'sa_module_3'},
        {'npoint': 8, 'radius': 0.8, 'nsample': 16, 'filters': [64,64,128], 'group_all': False, 'name': 'sa_module_4'}    
    ]
}

pointnet2_enc6 = {
    'sa_settings': [
        {'npoint': 32, 'radius': 0.1, 'nsample': 8, 'filters': [2,2,2], 'group_all': False, 'name': 'sa_module_1'},
        {'npoint': 16, 'radius': 0.2, 'nsample': 8, 'filters': [2,2,4], 'group_all': False, 'name': 'sa_module_2'},
        {'npoint': 8, 'radius': 0.4, 'nsample': 8, 'filters': [4,4,8], 'group_all': False, 'name': 'sa_module_3'},
        {'npoint': 4, 'radius': 0.8, 'nsample': 8, 'filters': [8,8,16], 'group_all': False, 'name': 'sa_module_4'}    
    ],
    "post_dropout": 0.2
}

pointnet2_enc7 = {
    'sa_settings': [
        {'npoint': 32, 'radius': 0.1, 'nsample': 8, 'filters': [2,2,4], 'group_all': False, 'name': 'sa_module_1'},
        {'npoint': 16, 'radius': 0.2, 'nsample': 8, 'filters': [4,4,8], 'group_all': False, 'name': 'sa_module_2'},
        {'npoint': 8, 'radius': 0.4, 'nsample': 8, 'filters': [8,8,16], 'group_all': False, 'name': 'sa_module_3'},
        {'npoint': 4, 'radius': 0.8, 'nsample': 8, 'filters': [16,16,32], 'group_all': False, 'name': 'sa_module_4'}
    ],
    "post_dropout": 0.2
}

gru_baseline = {
    "units"              : 256,
    "activation"         : "tanh",
    "kernel_initializer" : "glorot_uniform",
    "return_sequences"   : True,
    "return_state"       : False,
    "stateful"           : False
}

gru_baseline_copy = {
    "units"              : -1,
    "activation"         : "tanh",
    "kernel_initializer" : "glorot_uniform",
    "return_sequences"   : True,
    "return_state"       : False,
    "stateful"           : False
}

gru_pnet2 = {
    "units": -1,
        "activation": "tanh",
        "kernel_initializer": "glorot_uniform",
        "return_sequences": False,
        "return_state": False,
        "stateful": False,
        "dropout": 0.2,
        "kernel_regularizer": {
            "name": "l2",
            "value": 0.01
        },
        "recurrent_regularizer": {
            "name": "l2",
            "value": 0.01
        },
        "bias_regularizer": {
            "name": "l2",
            "value": 0.01
        },
        "activity_regularizer": {
            "name": "l2",
            "value": 0.01
        }
}

fc_baseline = {
    "units"              : [-1,100],
    "kernel_initializer" : "glorot_uniform"
}

pointnet1_3D_fcupconv = {
    "fc_args": {
        "units"              : [-1,100,300],
        "kernel_initializer" : "glorot_uniform"
    },
    "upconv_args": {
        'filters_dividers'    : [1, 2, 4, 8],
        'kernel_sizes'       : [[1,2,2], [1,3,3], [1,4,4], [1,5,5]],
        'strides'            : [[1,1,1], [1,1,1], [1,2,2], [1,2,2]],
        'padding'            : 'VALID',
        'batch_norm'         : False,
        'max_pool_size'      : 4,
        'max_pool_stride'    : 4,
        'max_pool_padding'   : 'SAME',
        "use_xavier"         : True,
        "stddev"             : 1e-3,
        "weight_decay"       : None,
        "activation_fn"      : "relu",
        "batch_norm_decay"   : 0.9,
    }
}

pointnet2_sem_seg_dec = {
    'fp_settings': [
        {'mlp': [256, 256], 'name': 'fp_module_0'},
        {'mlp': [256, 256], 'name': 'fp_module_1'},
        {'mlp': [256, 128], 'name': 'fp_module_2'},
        {'mlp': [128,128,128], 'name': 'fp_module_3'}    
    ]
}

pointnet2_sem_seg_dec_reduced = {
    'fp_settings': [
        {'mlp': [64, 64], 'name': 'fp_module_0'},
        {'mlp': [64, 64], 'name': 'fp_module_1'},
        {'mlp': [64, 32], 'name': 'fp_module_2'},
        {'mlp': [32,32,32], 'name': 'fp_module_3'}    
    ]
}

pointnet2_dec6 = {
    'fp_settings': [
        {'mlp': [16, 16], 'name': 'fp_module_0'},
        {'mlp': [16, 16], 'name': 'fp_module_1'},
        {'mlp': [16, 8], 'name': 'fp_module_2'},
        {'mlp': [8,8,8], 'name': 'fp_module_3'}    
    ],
    "out_points": 30,
    "append_xyz": False,
    "pre_dropout": 0.2,
    "post_dropout": 0.2
}

all_models = {
    "pointnet1_Conv3D": pointnet1_Conv3D,
    "pointnet1_Conv3D_short": pointnet1_Conv3D_short,
    "pointnet1_Conv2D": pointnet1_Conv2D,
    "pointnet2_sem_seg_enc" : pointnet2_sem_seg_enc,
    "pointnet2_sem_seg_enc_reduced": pointnet2_sem_seg_enc_reduced,
    "pointnet2_enc6": pointnet2_enc6,
    "pointnet2_enc7": pointnet2_enc7,
    "gru_baseline": gru_baseline,
    "gru_baseline_copy": gru_baseline_copy,
    "gru_pnet2": gru_pnet2,
    "fc_baseline": fc_baseline,
    "pointnet1_3D_fcupconv": pointnet1_3D_fcupconv,
    "pointnet2_sem_seg_dec": pointnet2_sem_seg_dec,
    "pointnet2_sem_seg_dec_reduced": pointnet2_sem_seg_dec_reduced,
    "pointnet2_dec6": pointnet2_dec6,
}
