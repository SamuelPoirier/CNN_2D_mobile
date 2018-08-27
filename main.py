# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 10:25:18 2018

@author: SAPOI22
"""

import os
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing_mod_MFCC import get_train_test_set
from conv_2d_mobile import conv_2d_mobile_model


def save_best_model(number):
    filepath="best_models//" + "weights_" + str(number) + ".hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    # check 5 epochs
    early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, mode='max') 
    callbacks_list = [checkpoint, early_stop]
    return callbacks_list


def build_CNN(random_state):
    DATA_PATH = os.path.abspath('C:\\Users\\Sapoi22\\Codes\\Models\\CNN_2D_mobile\\google_dataset')
    NUMPY_FILE_PATH = os.path.abspath('C:\\Users\\Sapoi22\\Codes\\Models\\CNN_2D_mobile\\google_dataset_numpy_13MFCC_16kHz_time_160')
    X_train, X_test, y_train, y_test = get_train_test_set(data_path=DATA_PATH, 
                                                          np_file_path=NUMPY_FILE_PATH, 
                                                          random_state=random_state)
    
    feature_dim1 = 160
    feature_dim2 = 13
    
    X_train = X_train.reshape(X_train.shape[0], feature_dim1, feature_dim2, 1)
    X_test = X_test.reshape(X_test.shape[0], feature_dim1, feature_dim2, 1)
    np.save('saved_samples\\training_samples.npy', X_train)
    np.save('saved_samples\\training_labels.npy', y_train)
    np.save('saved_samples\\test_samples.npy', X_test)
    np.save('saved_samples\\test_labels.npy', y_test)
    
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    number_of_word = np.ma.size(y_test_hot, axis=1)

    for model_number in range(1):
        model_CNN = conv_2d_mobile_model(time_size=feature_dim1, frequency_size=feature_dim2, num_classes=number_of_word)
        callbacks_list = save_best_model(model_number)
        model_CNN.fit(X_train, y_train_hot, X_test, y_test_hot, batch_size=10, epochs=20, callbacks=callbacks_list)

if __name__ == '__main__':
    build_CNN(10)