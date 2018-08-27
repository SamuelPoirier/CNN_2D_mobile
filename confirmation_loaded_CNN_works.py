# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:00:58 2018

@author: Sapoi22
"""
import os
import keras
from keras import backend as K
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.utils import to_categorical
from save_load_CNN_model import load_data


def load_CNN_model(path):
    with CustomObjectScope({'relu6':  keras.applications.mobilenet.relu6,
                            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(path)
    return model


if __name__ == '__main__':  
    loaded_data_dict = load_data('saved_samples\\')
    x_test = loaded_data_dict['test_samples']
    y_test = loaded_data_dict['test_labels']
    y_test_hot = to_categorical(y_test)
    model = load_CNN_model('best_models//weights_0.hdf5')
    path = os.path.abspath('C:\\Users\\Sapoi22\\Codes\\Models\\CNN_2D_mobile\\google_dataset')
    loss, acc = model.evaluate(x_test, y_test_hot, verbose=1)