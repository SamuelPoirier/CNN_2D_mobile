# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:00:58 2018

@author: Sapoi22
"""
import os
import keras
import numpy as np
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.models import Model
from keras.utils import to_categorical
from save_load_CNN_model import load_data

def load_CNN_model(path):
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(path)
    return model

def get_output_layer(loaded_model, samples, layer_wanted=-3):
    #https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    model = loaded_model
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(index=layer_wanted).output)
    for sample in samples:
        sample = sample.reshape(1, sample.shape[0], sample.shape[1], sample.shape[2])
        intermediate_output = intermediate_layer_model.predict(sample)
        layer_output = np.squeeze(intermediate_output)
        yield layer_output


if __name__ == '__main__':
    model = load_CNN_model('best_models//weights_0.hdf5')
    loaded_data_dict = load_data('saved_samples\\')
    x_train = loaded_data_dict['training_samples']
    y_train = loaded_data_dict['training_labels']
    y_train_hot = to_categorical(y_train)
    
    layers_wanted = [-1, -2, -3]
    for layer_wanted in layers_wanted:
        outputs_wanted = get_output_layer(model, x_train, layer_wanted=layer_wanted)
        outputs = [x for x in outputs_wanted]
        np.save('saved_samples\\outputs_of_layer' + str(layer_wanted) + '.npy', outputs)