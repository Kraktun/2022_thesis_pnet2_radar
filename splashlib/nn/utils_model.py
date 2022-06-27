import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from collections.abc import Mapping

from splashlib.nn import custom_losses

from splashlib.nn.keras_layers import *
from splashlib.nn.utils_layers import *
from splashlib.nn.nets.pointnet2 import pointnet2
from splashlib.nn.nets import pointnet


def model_from_builder(preproc_model_builder, model_builder):
    """
    Generate the model according to the values of preprocessing and main builders.
    """
    mb = model_builder 
    pmb = preproc_model_builder

    # apply preprocessing layers on the input data. 
    # Remember to call preproc_model_builder.apply_data_preproc(dataset) on the dataset
    # for both training and test data, as some operations must be applied directly to the dataset.
    preproc_model = _pre_model_from_builder(pmb)

    # ENCODER STRUCTURE
    assert mb.encoder_type in ['pointnet1', 'pointnet1_fc', 'pointnet2'], "Invalid encoder type"
    enc_args = mb.encoder_args.copy() # copy, as we modify them
    # input layer
    if mb.decoder_type == 'pointnet2_split':
        input_layer = preproc_model.outputs[0]
        preproc_input = preproc_model.inputs[0]
    else:
        input_layer = tf.keras.Input(shape=pmb.out_shape())

    if mb.encoder_type == 'pointnet1' or mb.encoder_type == 'pointnet1_fc':
        encoder = pointnet.get_encoder(in_layer=input_layer, enc_args=enc_args, preproc_model_builder=pmb)
    elif mb.encoder_type == 'pointnet1_fc':
        # add final fc layer
        encoder = pointnet.get_encoder_fc(in_layer=input_layer, enc_args=enc_args, preproc_model_builder=pmb)
    elif mb.encoder_type == 'pointnet2':
        encoder_states, encoder = pointnet2.Pointnet2Encoder(enc_args)(input_layer)
    
    # RNN STRUCTURE
    assert mb.rnn_type in ['gru', 'identity', 'identity_last_frame', 'unstacked_gru', 'concat_gru', 'unstacked_lstm', 'concat_lstm', 'cnn_sequence'], "Invalid RNN type"
    rnn_args = mb.rnn_args.copy()
    rnn_args = parse_regularizers(rnn_args)

    if mb.rnn_type == 'identity':
        rnn = IdentityLayer(rnn_type=mb.rnn_type, rnn_args=mb.rnn_args)(encoder)
    elif mb.rnn_type == 'identity_last_frame':
        rnn = IdentityLayer(rnn_type=mb.rnn_type, rnn_args=mb.rnn_args)(encoder)
        encoder_states = [encoder_states[-1]] 
    elif mb.rnn_type == 'cnn_sequence':
        rnn, encoder_states = CnnSequence(rnn_type=mb.rnn_type, rnn_args=mb.rnn_args)([encoder, encoder_states])
    else:
        
        if mb.rnn_type == 'unstacked_gru':
            rnn = UnstackedRnn(rnn_type='gru', rnn_args=rnn_args)(encoder)
        elif mb.rnn_type == 'concat_gru':
            rnn = ConcatRnn(rnn_type='gru', rnn_args=rnn_args)(encoder)
        elif mb.rnn_type == 'unstacked_lstm':
            rnn = UnstackedRnn(rnn_type='lstm', rnn_args=rnn_args)(encoder)
        elif mb.rnn_type == 'concat_lstm':
            rnn = ConcatRnn(rnn_type='lstm', rnn_args=rnn_args)(encoder)
        
        if not rnn_args['return_sequences']:
            # take last time step and reinsert in a list, to make the net compatible with return_sequences=True
            encoder_states = [encoder_states[-1]] 

    # DECODER STRUCTURE
    assert mb.decoder_type in ['fc_base', 'fc_upconv', 'pointnet2', 'pointnet2_split'], "Invalid decoder type"
    dec_args = mb.decoder_args.copy()

    if mb.decoder_type == 'fc_base':
        decoder = pointnet.get_decoder_fc(in_layer=rnn, dec_args=dec_args, preproc_model_builder=pmb)
    elif mb.decoder_type == 'fc_upconv':
        decoder = pointnet.get_decoder_upconv(in_layer=rnn, dec_args=dec_args, preproc_model_builder=pmb)
    elif mb.decoder_type == 'pointnet2':
        decoder, dec_outputs = pointnet2.Pointnet2Decoder(dec_args)([rnn, encoder_states])
    elif mb.decoder_type == 'pointnet2_split':
        decoder, dec_outputs = pointnet2.Pointnet2DecoderCore(dec_args)([rnn, encoder_states])
        markers_out = pointnet2.DecoderExtension(dec_args=dec_args, mode="supervised", name="MarkersDecoder")(decoder)
        cloud_out = pointnet2.DecoderExtension(dec_args=dec_args, mode="unsupervised", name="CloudDecoder")(decoder)
        decoder = [markers_out, cloud_out]

    if mb.decoder_type == 'pointnet2_split':
        input_model = preproc_input
    else:
        input_model = input_layer
    # learning model definition
    learn_model = Model(inputs=input_model, outputs=decoder, name="Learning_model")
    # build partial models to get output at specific points:
    if mb.encoder_type == "pointnet2":
        encoder = encoder_states # use states as output of encoder, it will also include the enc_layer
    encoder_model = Model(inputs=input_model, outputs=encoder, name="Encoder_model")
    rnn_model = Model(inputs=input_model, outputs=rnn, name="Rnn_model")
    # no need for Decoder_model as it is the same as learn_model
    
    # define loss, optimizer etc.
    def _get_loss_by_name(loss_name):
        my_loss = None
        if loss_name.lower() == 'chamfer':
            my_loss = custom_losses.chamfer_loss
        elif loss_name.lower() == 'hausdorff':
            my_loss = custom_losses.hausdorff_loss
        elif loss_name.lower() == 'squared_dist':
            my_loss = custom_losses.perpoint_squared_distance
        elif loss_name.lower() == 'selective_chamfer':
            my_loss = custom_losses.selective_chamfer_loss
        elif loss_name.lower() == 'selective_hausdorff':
            my_loss = custom_losses.selective_hausdorff_loss
        elif loss_name.lower() == 'selective_squared_dist':
            my_loss = custom_losses.selective_perpoint_squared_distance
        elif loss_name.lower() == 'selective_dist':
            my_loss = custom_losses.selective_perpoint_distance
        elif loss_name.lower() == 'manhattan_dist':
            my_loss = custom_losses.perpoint_manhattan_distance
        elif loss_name.lower() == 'selective_manhattan_dist':
            my_loss = custom_losses.selective_perpoint_manhattan_distance
        elif loss_name.lower() == 'cosine_dist':
            my_loss = custom_losses.perpoint_cosine_distance
        elif loss_name.lower() == 'selective_cosine_dist':
            my_loss = custom_losses.selective_perpoint_cosine_distance
        elif loss_name is not None:
            my_loss = loss_name # accpet other strings (e.g. mse)
        return my_loss
    
    if mb.decoder_type == 'pointnet2_split':
        my_loss = {
            "MarkersDecoder": _get_loss_by_name(dec_args['model_dec_supervised']['loss']),
            "CloudDecoder": _get_loss_by_name(dec_args['model_dec_unsupervised']['loss'])
        }
        loss_weights = {
            "MarkersDecoder": dec_args['model_dec_supervised']['loss_weight'],
            "CloudDecoder": dec_args['model_dec_unsupervised']['loss_weight']
        }
    else:
        my_loss = _get_loss_by_name(mb.loss)
        loss_weights = None
    
    my_optimizer = 'rmsprop'
    if isinstance(mb.optimizer, Mapping): # check if it's a dict
        if mb.optimizer['name'].lower() == 'adam':
            my_optimizer = tf.keras.optimizers.Adam(**mb.optimizer)
        elif mb.optimizer['name'].lower() == 'adadelta':
            my_optimizer = tf.keras.optimizers.Adadelta(**mb.optimizer)
        elif mb.optimizer['name'].lower() == 'sgd':
            my_optimizer = tf.keras.optimizers.SGD(**mb.optimizer)
        elif mb.optimizer['name'].lower() == 'rmsprop':
            my_optimizer = tf.keras.optimizers.RMSprop(**mb.optimizer)
        elif mb.optimizer['name'].lower() == 'adagrad':
            my_optimizer = tf.keras.optimizers.Adagrad(**mb.optimizer)
        elif mb.optimizer['name'].lower() == 'adamax':
            my_optimizer = tf.keras.optimizers.Adamax(**mb.optimizer)
        elif mb.optimizer['name'].lower() == 'nadam':
            my_optimizer = tf.keras.optimizers.Nadam(**mb.optimizer)
        elif mb.optimizer['name'].lower() == 'Ftrl':
            my_optimizer = tf.keras.optimizers.Ftrl(**mb.optimizer)
        else:
            my_optimizer = mb.optimizer['name']
    elif mb.optimizer is not None:
        my_optimizer = mb.optimizer
    
    my_metrics = None
    if mb.metrics is not None:
        my_metrics = mb.metrics
    
    # combine preprocessing and learning models
    if mb.decoder_type == 'pointnet2_split':
        full_model = Model(inputs=preproc_input, outputs=decoder)
    else:
        full_model = Sequential([preproc_model, learn_model])
    # compile
    full_model.compile(optimizer=my_optimizer, loss=my_loss, metrics=my_metrics, loss_weights=loss_weights)
    
    # return all models
    return full_model, (preproc_model, encoder_model, rnn_model, learn_model)


def _pre_model_from_builder(model_builder):
    
    mb = model_builder
    
    if model_builder.model_type == 'generic':
        input_layer = tf.keras.Input(shape=mb.in_shape())
        mid_layer = input_layer
        if mb.random_shift:
            mid_layer = RandomShift(prob=mb.random_shift_prob, interval=mb.random_shift_interval)(mid_layer)
        if mb.random_permutation:
            mid_layer = RandomPermutation(prob=mb.random_permutation_prob, axis=mb.random_permutation_axis)(mid_layer)
        if mb.random_shuffle:
            mid_layer = RandomCloudPermutation(prob=mb.random_shuffle_prob)(mid_layer)
        if mb.random_rotation:
            mid_layer = RandomRotation(prob=mb.random_rotation_prob, ang_intervals=mb.random_rotation_angles)(mid_layer)
        if mb.add_channel:
            mid_layer = layers.Reshape(mid_layer.shape[1:] + (1,))(mid_layer) # remove batch + add one at the end
        output_layer = mid_layer
        
    preproc_model = Model(inputs=input_layer, outputs=output_layer, name="Preprocessing_Model")
    return preproc_model
