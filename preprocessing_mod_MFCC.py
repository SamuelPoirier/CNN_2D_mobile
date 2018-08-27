# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:01:02 2018

@author: Sapoi22
"""

import os
import numpy as np
#import librosa
#from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def wav2mfcc(file_path, max_pad_len=160, zero_pad_sym=False):
    """
    Input: Folder Path
    Output: mfcc (log_energy replace the first MFCC and added to 2-13 MFCCs; 11 sets of (1 log_energy + 12 MFCCs))
    """
    wave, rate = librosa.core.load(file_path, mono=True, sr=None)
    #wave = wave[::3]
    mfcc_matrix = mfcc(wave, samplerate=rate, nfilt=40, numcep=40, appendEnergy=True, nfft=1103) # NFFT = 1103, maybe not good
    pad_width = max_pad_len - mfcc_matrix.shape[0]
    pad_after = pad_width//2
    
    if zero_pad_sym:
        if pad_after != 0 and pad_width%(2*pad_after) == 0:
            pad_before = pad_after
        else:
            pad_before = pad_after + 1
        mfcc_matrix = np.pad(mfcc_matrix, pad_width=((pad_before, pad_after), (0, 0)), mode='constant') #pad_width is a tuple of (n_before, n_after) for each dimension
    else:
        mfcc_matrix = np.pad(mfcc_matrix, pad_width=((pad_width, 0), (0, 0)), mode='constant') #pad_width is a tuple of (n_before, n_after) for each dimension
    assert mfcc_matrix.shape[0] == max_pad_len
    return mfcc_matrix


def get_labels(path):
    """
    Input: Folder Path
    Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
    """
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices


def save_data_to_array(path, save_path, max_pad_len=160, zero_pad_sym=False):
    labels, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []
        
        wavfiles = [path + '\\' + label + '\\' + wavfile 
                    for wavfile in os.listdir(path + '\\' + label)]
        
        for wavfile in wavfiles:
            mfcc = wav2mfcc(wavfile, max_pad_len=max_pad_len, zero_pad_sym=zero_pad_sym)
            mfcc_vectors.append(mfcc)
        np.save(os.path.join(save_path,label + '.npy'), mfcc_vectors)
        
        
def get_balanced_sets(data_path, np_file_path, split_ratio, random_state):
    # Get available labels
    labels, indices = get_labels(data_path)
    min_train_length = 9999
    
    # Safety for different number of examples per word
    for label in labels:
        train_length = np.floor((np.load(np_file_path + '\\' + label + '.npy')).shape[0]*split_ratio)
        if train_length < min_train_length:
            min_train_length = train_length
    min_train_length = int(min_train_length)
    min_test_length = int(np.floor(min_train_length/split_ratio*(1-split_ratio)))
    
    for i, label in enumerate(labels):
        x = np.load(np_file_path + '\\' + label + '.npy')
        y = np.full(x.shape[0], fill_value=i) # Labels for 3 words : 0, 1, 2. There are as many labels 0 as there is word examples for the first word
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=split_ratio, 
                                                            random_state=random_state, shuffle=True)
        X_train = X_train[:min_train_length,:,:]
        X_test = X_test[:min_test_length,:,:]
        y_train = y_train[:min_train_length,]
        y_test = y_test[:min_test_length,]
        if i == 0:
            x_train_set = X_train
            x_test_set = X_test
            y_train_set = y_train
            y_test_set = y_test
        else:
            x_train_set = np.vstack((x_train_set, X_train))
            x_test_set = np.vstack((x_test_set, X_test))
            y_train_set = np.vstack((y_train_set, y_train))
            y_test_set = np.vstack((y_test_set, y_test))
    y_train_set = np.reshape(y_train_set, (y_train_set.shape[0]*y_train_set.shape[1]))
    y_test_set = np.reshape(y_test_set, (y_test_set.shape[0]*y_test_set.shape[1]))
    

    assert x_train_set.shape[0] == y_train_set.shape[0]

    return x_train_set, x_test_set, y_train_set, y_test_set


def get_train_test_set(data_path, np_file_path, split_ratio=0.7, random_state=42):
    x_train_set, x_test_set, y_train_set, y_test_set = get_balanced_sets(data_path, np_file_path, split_ratio, random_state)
    x_train_set, y_train_set = shuffle(x_train_set, y_train_set, random_state=random_state)
    x_test_set, y_test_set = shuffle(x_test_set, y_test_set, random_state=random_state)
    return x_train_set, x_test_set, y_train_set, y_test_set
# TODO Make a generator for the train/test sets
    
if __name__ == "__main__":
    """
    Old method:
    script_dir = os.path.abspath('U:\Code_initial\Google_dataset\google_dataset\one')
    rel_path = '00f0204f_nohash_0.wav'
    abs_file_path = os.path.join(script_dir, rel_path)
    y = wav2mfcc(abs_file_path)
    """

    DATA_PATH = os.path.abspath('C:\\Users\\Sapoi22\\Codes\\Dataset\\Compressed_Norm_no_silence_Sams_record')
    SAVE_PATH = 'C:\\Users\\Sapoi22\\Codes\\Models\\CNN_2D_mobile\\comp_norm_no_silence_Sams_40_MFCC'
    save_data_to_array(DATA_PATH, SAVE_PATH, zero_pad_sym=True)
    