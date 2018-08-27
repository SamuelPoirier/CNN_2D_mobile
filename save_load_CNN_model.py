import os
import numpy as np
from keras.models import model_from_json, Model
from keras import backend as K


def save_CNN_model(model_to_save):
    model_json = model_to_save.model.to_json()
    with open("CNN_model.json", "w") as json_file:
        json_file.write(model_json)
    model_to_save.model.save_weights("model_weights.h5")
    print("Saved model to disk")


def load_CNN_model(saved_model_json, saved_weights):
    with open(saved_model_json, 'r') as f:
        loaded_model = model_from_json(f.read())
    loaded_model.load_weights(saved_weights)
    return loaded_model


def load_data(files_directory):
    #files_directory = 'saved_samples\\'
    files = os.listdir(files_directory)
    loaded_data = {}
    for file in files:
        path = files_directory + file
        print('Loaded:' + path)
        loaded_data[file[:-4]] = np.load(path)
    return loaded_data


#def get_output_layer(loaded_model, samples, layer_wanted=-3,):
#    #https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
#    model_input = loaded_model.input
#    layer_outputs = loaded_model.layers[layer_wanted].output
#    functor = K.function([model_input] + [K.learning_phase()], (layer_outputs,))
#    #functor = K.function([model_input, K.learning_phase()], [layer_outputs])
#    for sample in samples:
#        sample = sample.reshape(1, sample.shape[0], sample.shape[1], sample.shape[2])
#        layer_output = functor([sample, 1.])
#        layer_output = np.squeeze(layer_output)
#        yield layer_output
        
def get_output_layer(loaded_model, samples, layer_wanted='layer_wanted'):
    #https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    model = loaded_model
    layer_name = layer_wanted
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    for sample in samples:
        sample = sample.reshape(1, sample.shape[0], sample.shape[1], sample.shape[2])
        intermediate_output = intermediate_layer_model.predict(sample)
        layer_output = np.squeeze(intermediate_output)
        yield layer_output
        
#def get_layer(loaded_model, layer_wanted=-3):
    #https://groups.google.com/forum/#!topic/keras-users/nSwjZu6ygA8
    
#    model = loaded_model
#    layer = model.layers[-3]  # Adjust here to the right depth. 
#    print(layer)
#    tensor = layer.get_output_at(0) 
#    print(tensor)
#    f = K.function(model.inputs + [K.learning_phase()], (tensor,)) 
#    print(f)
#    return f





#https://code.oursky.com/tensorflow-svm-image-classifications-engine/

#k-fold
#https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
