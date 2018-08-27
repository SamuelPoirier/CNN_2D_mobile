# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:42:33 2018

@author: SAPOI22
"""

import keras
from keras import backend as K
from keras.layers import (Input, Reshape, Conv2D, Lambda, BatchNormalization, 
                          Activation, Dense, Dropout, GlobalAveragePooling2D)
from keras.models import Model


#def preprocess_raw(x):
#  # x = K.pow(10.0, x) - 1.0
#  return x


class conv_2d_mobile_model:
    """
      https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/models.py#L165-L270  # noqa
      Args:
        input_size: How big the input vector is.
        num_classes: How many classes are to be recognized.
      Returns:
        Compiled keras model
    """
    
    def __init__(self, time_size=160, frequency_size=13, num_classes=8):
        #self.input_size = input_size
        self.time_size = time_size
        self.frequency_size = frequency_size
        self.num_classes = num_classes
    
    def preprocess(self, x):
        #print(*args)
        x = (x + 0.8) / 7.0
        x = K.clip(x, -5, 5)
        return x
    
    def relu6(self, x):
        return K.relu(x, max_value=6)
        
    def _conv_bn_relu6(self, x, num_filter, kernel=(3, 3), strides=(1, 1)):
        x = Conv2D(num_filter, kernel, padding='same',
                   strides=strides)(x)
        x = BatchNormalization()(x)
        x = Activation(self.relu6)(x)
        return x
    
    def create_model(self):
#        TODO(see--): Allow variable sized frequency_size & time_size
#        time_size = 160
#        frequency_size = 100
#        
#        input_layer = Input(shape=[16000])
#        x = input_layer
#        x = Reshape([time_size, frequency_size, 1])(x)
#        Preprocess = Lambda(self.preprocess)
#        x = Preprocess(x)
#        print(x)
                
        
        input_layer = Input(shape=[self.time_size, self.frequency_size, 1])
        print(input_layer)
        x = input_layer
        #x = Reshape([self.time_size, self.frequency_size, 1])(x)
        #Preprocess = Lambda(self.preprocess)
        #x = Preprocess(x)

        x = self._conv_bn_relu6(x, 32, strides=2)  # (49, 20)
        x = self._conv_bn_relu6(x, 32)  # (49, 20)
        x = Dropout(0.05)(x)
        x = self._conv_bn_relu6(x, 64, strides=2)  # (25, 10)
        x = self._conv_bn_relu6(x, 64)  # (25, 10)
        x = Dropout(0.05)(x)
        x = self._conv_bn_relu6(x, 128, strides=2)  # (13, 5)
        x = self._conv_bn_relu6(x, 128)  # (13, 5)
        x = Dropout(0.05)(x)
        x = self._conv_bn_relu6(x, 256, strides=2)  # (7, 3)
        x = self._conv_bn_relu6(x, 256)  # (7, 3)
        x = Dropout(0.05)(x)
        
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.1)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(input_layer, x, name='conv_2d_mobile')
        model.compile(
            optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.95),
            loss=keras.losses.categorical_crossentropy,
            metrics=[keras.metrics.categorical_accuracy])
        return model
    
    def fit(self, X_train, y_train_hot, X_test, y_test_hot, batch_size=10, epochs=50, callbacks=None):
        model = self.create_model()
        model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs,
                      verbose=1, validation_data=(X_test, y_test_hot), callbacks=callbacks)