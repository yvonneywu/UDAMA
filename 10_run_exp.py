#!/usr/bin/env python
# coding: utf-8


#import keras
from tensorflow import keras 
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.layers import CuDNNGRU 

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

import glob
import pandas as pd
import time
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from utils import min_max_scaling, error_metrics
from utils import *
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import argparse



from tensorflow.keras.models import Sequential, load_model, Model, model_from_json
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
import glob

from sklearn.model_selection import KFold

#set seed
import random 
from numpy.random import seed
seed(1)
tf.random.set_seed(2) 

print('=======================START ===========================')
#normalize seeting
met_indices = [19,20,21,22]
daily_count_indices = [9,10,11]
feat_indices = [12,13,14,15,16,17,18,23,24,25]
meta_indices = [0,1,2,3,4,5,6,7,8] # 26th features is the id ,demographics data 


parser = argparse.ArgumentParser(description='NN training.')
parser.add_argument('-r', '--ratio', metavar='number', default=1, type=int) # add different scale of silver samples
parser.add_argument('-f', '--fold', metavar='number', default=1, type=int) # add different fold_number
parser.add_argument('-a1', '--a1', metavar='number', default=1, type=float) 
parser.add_argument('-a2', '--a2', metavar='number', default=1, type=float) 
batch_size = 8

args = parser.parse_args()
print(args.a1, args.a2)

# Create a folder and store the model
model_time = time.strftime("%Y%m%d-%H%M%S") #use timestamp as folder name...
path = 'adaptation/%s/'%model_time
os.makedirs(os.path.dirname('./adaptation/%s/'%model_time))
print('=============Model time is:', model_time)


#example data shape is 
# (2, 600, 10) (2,) (2, 9)

f_X_test = np.load("/example_data/x_test.npy")
f_y_test = np.load("/example_data/y_test.npy")
f_y_test = f_y_test[:,0].astype('float')
f_X_test_demo = np.load("/example_data/x_test_demo.npy")
f_X_test_demo = f_X_test_demo[:,:9].astype('float')

X = np.load("/example_data/e_x.npy")
y = np.load("/example_data/e_y.npy")
print(X.shape,y.shape)


# add same silver-samples to each fold
list = [x for x in range(len(y))]
seed_value = random.randrange(22222)
index = random.sample(list, args.ratio) 
print('======================Seed value is:', seed_value)
print('======================Silver samples index is:', index)
    
# load pre-trained model
model_time = '20221025-140338'

model_pre = model_from_json(open('./example_model/'+ model_time +'/model_architecture.json').read())#,custom_objects={'Attention': Attention}) #custom Attention layer) 
                                     #,custom_objects={'exp': Activation(exponential)}) #custom K.exp activation
                                     #,custom_objects={'Attention': Attention}) #custom Attention layer
files = glob.glob('./example_model/'+model_time+'/*.hdf5')
weights = sorted(files, key=lambda name: float(name[56:-5]))[0]
print ("=============Best model loaded:", weights)
model_pre.load_weights(weights) 
model_pre.compile(loss="mse", optimizer="adam")
# model_pre.summary()

# Define the k-fold Cross Validator
print ("=============Start CV training with added silver samples number is:", args.ratio)
kfold = KFold(n_splits=args.fold, random_state = 42, shuffle=True) #80%-20% splitation
fold_no = 1
result = []

for train, test in kfold.split(X,y):
    print('===============================================================')
    print(f'Training for fold {fold_no} ...')

    input_data = X.copy()
    targets_data = y.copy()
    X_train = min_max_scaling(input_data[train][:,:,feat_indices],feat_indices)
    X_test = min_max_scaling(input_data[test][:,:,feat_indices],feat_indices)

    X_train_demo= scaler.fit_transform(input_data[train][:,0,meta_indices]) #this should be the train set
    X_test_demo= scaler.fit_transform(input_data[test][:,0,meta_indices]) 

    y_train = targets_data[train]
    y_test = targets_data[test]

    #####################prepare silver-dataset####################
    silver_train, silver_demo,y_silver = f_X_test[index],f_X_test_demo[index],f_y_test[index]
    #add two domain labels 
    y_silver = add_whole_domain(y_silver, 0)
    y_gold = add_whole_domain(y_train, 1)

    data = np.concatenate((silver_train,X_train),axis = 0) 
    data_demo = np.concatenate((silver_demo,X_train_demo),axis = 0)
    labels = np.concatenate((y_silver,y_gold),axis = 0)

    data, data_valid,data_demo, data_demo_valid, labels, labels_valid = train_test_split(data,data_demo, labels, test_size=0.1, shuffle= True, random_state= 12312)
    print(data.shape,data_demo.shape,labels.shape)
    dataset = create_dataset(data, data_demo, labels, batch_size)
    val_dataset = create_dataset(data_valid, data_demo_valid, labels_valid, batch_size)


    encoder = Model(inputs = model_pre.input, outputs = model_pre.get_layer('concatenate').output)
    encoder.get_layer('bidirectional').trainable = False
    encoder.get_layer('batch_normalization').trainable = False
    encoder.get_layer('dense').trainable = False

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

        #add classification/regression dense layer     
        final = keras.layers.Dense(1)(x)
        clf = keras.Model(inputs, outputs=[final]) 

        return clf


    d_categorical = make_categorical_discriminator(encoder) 
    d_distribution = make_distribution_discriminator(encoder)
    predictor = make_predictor(encoder)

    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
    mse = keras.losses.MeanSquaredError()

    domain_optimizer = keras.optimizers.Adam(1e-4)
    label_optimizer = keras.optimizers.Adam(1e-3)
    categorical_optimizer = keras.optimizers.Adam(1e-4)

    train_acc_metric = keras.metrics.BinaryAccuracy()
    val_loss_metric = keras.metrics.MeanSquaredError()
    
    @tf.function
    def train_step(x_train, y_domain_categorical, y_domain_distribution, y_label):
        with tf.GradientTape() as d_tape, tf.GradientTape() as c_tape, tf.GradientTape() as l_tape:
            categorical_output = d_categorical(x_train, training=True)
            lcse = gaussian_nll(y_domain_categorical, categorical_output)

            domain_output = d_distribution(x_train, training=True)
            lnll = gaussian_nll(y_domain_distribution, domain_output)

            label_output = predictor(x_train, training=True)
            lmse = mse(y_label, label_output)

            loss =  lmse - args.a1 * lnll - args.a2 * lcse ##original

        #gradients
        categorical_grads = c_tape.gradient(lcse, d_categorical.trainable_weights)
        categorical_optimizer.apply_gradients(zip(categorical_grads, d_categorical.trainable_weights))

        domain_grads = d_tape.gradient(loss, d_distribution.trainable_weights)
        domain_optimizer.apply_gradients(zip(domain_grads, d_distribution.trainable_weights))

        label_grads = l_tape.gradient(loss, predictor.trainable_weights + encoder.trainable_weights)
        label_optimizer.apply_gradients(zip(label_grads, predictor.trainable_weights  + encoder.trainable_weights)) 


        return lcse,lnll, lmse, loss

    @tf.function
    def test_step(x_val,y_domain,y_label):    
        logits = predictor(x_val, training=False)
        val_loss_metric.update_state(y_label, logits)
    

    
    epochs = 100
    epoch_l1 = []
    epoch_l2 = []
    epoch_l3 = []
    epoch_acc = []
    epoch_loss = []
    epoch_val_loss = []
    patience = 10
    wait = 0
    best = 0 

    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start = time.time()
        b_l1 = []
        b_l2 = []
        b_l3 = []
        b_loss = []
        #change to batch
        for batch, (x_batch, y_batch) in enumerate(dataset):
            lcse,lnll, lmse, loss = train_step(x_batch,np.array(y_batch[:,1]).astype('float32').reshape((-1,1)), np.array(y_batch[:,2]).astype('float32').reshape((-1,1)) , np.array(y_batch[:,0]).astype('float32').reshape((-1,1)))

        # for batch, data in enumerate(dataset):
        #     l1,l2,loss = train_step(data[0], np.array(data[1][0][1]).astype('float32').reshape((-1,1)), np.array(data[1][0][0]).astype('float32').reshape((-1,1)))   #each batch...
            b_l1.append(lcse.numpy())
            b_l2.append(lnll.numpy())
            b_l3.append(lmse.numpy())
            b_loss.append(loss.numpy())

        epoch_l1.append(np.mean(b_l1))
        epoch_l2.append(np.mean(b_l2))
        epoch_l3.append(np.mean(b_l3))
        epoch_loss.append(np.mean(b_loss))

        print("cse: " , np.mean(b_l1), end = " | ")
        print("NLL: " , np.mean(b_l2), end = " | ")
        # print("NLL: " , (np.mean(b_l1) - min(b_l1))/(max(b_l1)-min(b_l1)), end = " | ")
        print("mse: " , np.mean(b_l3), end = " | ")
        print("total_loss: " , np.mean(b_loss), end = " | ")
        

        for j, d_val in enumerate(val_dataset):        
            test_step(d_val[0], np.array(d_val[1][0][1]).astype('float32').reshape((-1,1)), np.array(d_val[1][0][0]).astype('float32').reshape((-1,1)))   #each batch...)

        val_loss = val_loss_metric.result()
        epoch_val_loss.append(val_loss.numpy())
        print("val_loss: " , val_loss.numpy(), end = " | ")

        val_loss_metric.reset_states()

        stopEarly = Callback_EarlyStopping(epoch_val_loss, min_delta=0.1, patience = 10)
        if stopEarly:
            print("Callback_EarlyStopping signal received at epoch= %d/%d"%(epoch,epochs))
            print("Terminating training ")
            break


    predictor.save(path + 'fold_'+str(fold_no))

    predicted = predictor.predict([X_test, X_test_demo]) 
    predicted = np.squeeze((predicted))
    mse, rmse, mae, r2 =  error_metrics(y_test.astype('float'),predicted) #only calculate error on the HR (not HRV)
    corr = np.corrcoef(y_test, predicted)[0][1]
    
    tfr = [mse, rmse, mae, r2,corr]
    result.append(tfr)
    
    print ("\nMSE:", round(mse,3), 
               "\nRMSE", round(rmse,3),
               "\nMAE", round(mae,3), 
               "\nRˆ2", round(r2,3),
                "\nCorrelation", round(corr,3)
              )
    
    # #save the results to csv
    predicted_vs_truth_test = pd.DataFrame(np.column_stack((predicted,y_test)), columns=['predicted', 'truth'])
    predicted_vs_truth_test.to_csv(path + 'DA_Distribution_'+str(fold_no)+'.csv')

    fold_no = fold_no + 1

###display final average results
print ("=============Final result=============")
# print('=============Batch_size is:', batch_size)
result_2 = np.array(result)
np.save(path+"DA_Distribution_metrics.npy",result_2)


print ("AVG_MSE:", round(np.mean(result_2[:,0]),3), 
           "\nAVG_RMSE", round(np.mean(result_2[:,1]),3),
           "\nAVG_MAE", round(np.mean(result_2[:,2]),3), 
           "\nAVG_Rˆ2", round(np.mean(result_2[:,3]),3),
           "\nAVG_Correlation", round(np.mean(result_2[:,4]),3)
          )

print ("std_MSE:", round(np.std(result_2[:,0]),3), 
           "\nstd_RMSE", round(np.std(result_2[:,1]),3),
           "\nstd_MAE", round(np.std(result_2[:,2]),3), 
           "\nstd_Rˆ2", round(np.std(result_2[:,3]),3),
           "\nstd_Correlation", round(np.std(result_2[:,4]),3)
          )

print('=======================End ===========================')






