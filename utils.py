#!/usr/bin/env python
# coding: utf-8



from tensorflow import keras 
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.layers import CuDNNGRU #solve -->cannot import name 'CuDNNGRU' from 'keras.layers'

from tensorflow.keras.layers import Dense, Embedding, Masking, TimeDistributed, Dropout, Flatten, LSTM, GRU, Bidirectional, Activation, RepeatVector, InputLayer, Conv1D, Input, Lambda, MaxPooling1D,GlobalAveragePooling1D, multiply, Reshape, Lambda, Permute
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import Layer
#from tensorflow.keras.engine.topology import Layer
from tensorflow.keras import initializers, regularizers, constraints
import tensorflow.keras.backend as K
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
import time
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt 
# get_ipython().run_line_magic('matplotlib', 'inline')
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
import seaborn as sns
sns.set_context("poster")
import glob
import pandas as pd
import random
from numpy import random as nr


from tensorflow.keras.models import Sequential, load_model, Model, model_from_json
from tensorflow.keras.layers import BatchNormalization




def error_metrics(test, predicted):
    mse =  mean_squared_error(test, predicted) #MSE
    rmse =  sqrt(mean_squared_error(test, predicted)) #RMSE
    mae = mean_absolute_error(test, predicted) #MAE
    r2 = r2_score(test, predicted) #rˆ2
    return mse, rmse, mae, r2


def min_max_scaling(array,indices):
    from sklearn import preprocessing #min-max scaling row-wise (also split per feature)
    all_arrays = []
    
    for i in range(0,len(indices)):
        all_arrays.append(preprocessing.minmax_scale(array[:,:,i].T).T)
        
        #hacky but it's the only way to append here
    features_tensors =np.dstack([all_arrays[0],
                                 all_arrays[1],
                                 all_arrays[2],
                                 all_arrays[3],
                                 all_arrays[4],
                                 all_arrays[5],
                                 all_arrays[6],
                                 all_arrays[7],
                                 all_arrays[8],
                                 all_arrays[9],
                                #  all_arrays[10],
                                #  all_arrays[11],
                                #  all_arrays[12],
                                #  all_arrays[13],
                                #  all_arrays[14],
                                #  all_arrays[15],
                                #  all_arrays[16],
                                ]) #stack them again (2D to 3D) and test that max is 1 and min is 0
    return features_tensors
        

def gaussian_nll(ytrue, ypreds):

    
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]
    
    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)

#take part of the silver data as adversal info

###change here
def make_silver_random(X,d,y,ratio):
    list = [x for x in range(len(y))]
    #import seed_value
    seed_value = random.randrange(22222)
    index = random.sample(list, ratio) 
    # index = np.random.choice(len(y), size = ratio, replace = True)
    print('======================Seed value is:', seed_value)
    print('======================Silver samples index is:', index)
    silver_train = X[index]
    silver_demo = d[index]
    y_silver = y[index]


    return silver_train, silver_demo,y_silver

def make_silver(X,d,y,ratio):
    silver_train = X[0:ratio]
    silver_demo = d[0:ratio]
    y_silver = y[0:ratio]
    return silver_train, silver_demo,y_silver

def add_whole_domain(y,domain):
    #fisrt add categorical label
    if domain == 1:
        new_y = np.c_[y, np.ones(len(y)), nr.normal(loc=np.mean(y), scale=np.std(y), size=(len(y)))]
    else:
        new_y = np.c_[y, np.zeros(len(y)), nr.normal(loc=np.mean(y), scale=np.std(y), size=(len(y)))]    
    return new_y


    #then add distribution label

    #concatenate together 
    return new_y

def add_domain(y,domain):
    if domain == 1:
        new_y = np.c_[y,np.ones(len(y))]
    else:
        new_y = np.c_[y,np.zeros(len(y))]    
    return new_y

def add_domain_distribution(y):
    #concatenate original labels and new distribution
    new_y = np.c_[y, nr.normal(loc=np.mean(y), scale=np.std(y), size=(len(y)))]
   
    return new_y

def create_dataset(time_tensor, meta_tensor,label, batch_size):
    input_set = tf.data.Dataset.from_tensor_slices((time_tensor, meta_tensor))
    output_set = tf.data.Dataset.from_tensor_slices(label)
    # Create Dataset pipeline
    # !!! do not need .repeat()
    input_set = input_set.batch(batch_size)#.repeat()
    output_set = output_set.batch(batch_size)#.repeat()
    
    # Group the input and output dataset
    dataset = tf.data.Dataset.zip((input_set, output_set))
    
    return dataset


def Callback_EarlyStopping(LossList, min_delta, patience):
    #No early stopping for 2*patience epochs 
    if len(LossList)//patience < 2 :
        return False
    #Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
    mean_recent = np.mean(LossList[::-1][:patience]) #last
    #you can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_previous) #abs change

    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    if delta_abs < min_delta :
        print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
        print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False

def make_categorical_discriminator(encoder):
        time_input = Input(shape=(600, 10), name='time_tensors')
        meta_input = Input(shape=(9,), name='meta_tensors')
        inputs=[time_input, meta_input]
        x = encoder(inputs)
        #add classification dense layer     
        final = keras.layers.Dense(1, activation='sigmoid')(x)
        clf = keras.Model(inputs, outputs=[final]) 
        return clf

def make_distribution_discriminator(encoder):
    time_input = Input(shape=(600, 10), name='time_tensors')
    meta_input = Input(shape=(9,), name='meta_tensors')
    inputs=[time_input, meta_input]
    x = encoder(inputs)  
    final = keras.layers.Dense(1, activation='sigmoid')(x)
    clf = keras.Model(inputs, outputs=[final]) 
    return clf

def make_predictor(encoder):
    time_input = Input(shape=(600, 10), name='time_tensors')
    meta_input = Input(shape=(9,), name='meta_tensors')
    inputs=[time_input, meta_input]
    x = encoder(inputs)   
    final = keras.layers.Dense(1)(x)
    clf = keras.Model(inputs, outputs=[final]) 
    return clf

def create_model(dim_shape, modality,timestamp):

    if modality == 1: #timeseriesed
        # define model
        model = keras.Sequential(name="my_sequential")
        model.add(Input(shape=(timestamp, dim_shape), name='tensors')) #150,17 -->X_train.shape[1], X_train.shape[2]
        # add Convolutional layers
        model.add(Conv1D(filters=256, kernel_size=3, padding='valid', strides=1, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.01)))
        model.add(Dropout(rate=0.33))
        model.add(Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu'))
        model.add(Dropout(rate=0.33))
        model.add(Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu'))
        model.add(Dropout(rate=0.33))
        model.add(Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu'))
        model.add(Dropout(rate=0.33))

        # add RNN layers
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Dropout(rate=0.2))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Dropout(rate=0.2))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(Bidirectional(GRU(128, return_sequences=True)))

        # add pooling layer
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='linear'))

    if modality == 2: ##time + meta
        #time
        time_input = Input(shape=(timestamp, dim_shape), name='time_tensors')
#         x = Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu',
#                    kernel_regularizer=keras.regularizers.l2(0.01))(time_input)
#         x = Dropout(0.2)(x)
#         x = Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu')(x)
#         x = Dropout(0.2)(x)
#         x = Conv1D(filters=128, kernel_size=3, padding='valid', strides=1, activation='relu')(x)
#         x = Dropout(0.2)(x)
        
        x = Bidirectional(GRU(32, return_sequences=True))(time_input)
        x = Bidirectional(GRU(32, return_sequences=True))(x)
        x = GlobalAveragePooling1D()(x)

        #meta
        meta_input = Input(shape=(9,), name='meta_tensors') #
        y = BatchNormalization()(meta_input)
        y = Dense(128, activation='relu')(y)
        y = Dense(128, activation='relu')(y)
        y = Dropout(0.33)(y)

        output = keras.layers.concatenate([x,y])
        final = Dense(1, activation='linear')(output)
        model = Model(inputs=[time_input, meta_input], outputs=[final])

        # summary and complie the model
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    return model
