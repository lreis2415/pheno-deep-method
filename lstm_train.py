# -*- coding: utf-8 -*-
import keras
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Embedding,Dropout,Activation,Dense
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import tensorflow.keras.backend as K

# self defined callback
# for function trainModel 
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.4
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    if lrate >=0.0001:
        return lrate
    else:
        return 0.0001
# self-defined loss function
# for function trainModel     
def binary_focal_loss(gamma=2, alpha=0.5):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed

# the function for training the lstm model 
# @feature_sample0_train: sample for non-rice paddy with feature data,
#                         the output of function getTrainSample (from sampling.py),
#                         as type of two-dimension numpy array [feature_length,sample_num]
# @feature_sample1_train: sample for rice paddy with feature data,
#                         the output of function getTrainSample (from sampling.py),
#                         as type of two-dimension numpy array [feature_length,sample_num]
# @feature_sample0_val: the validation sample for non-rice paddy with feature data,
#                       as type of two-dimension numpy array [feature_length,sample_num]
# @feature_sample1_val: the validation sample for rice paddy with feature data,
#                       as type of two-dimension numpy array [feature_length,sample_num]
# @data_dim: the number of spectral bands and remote sensing indexes used to create feature data
# @timesteps: the number of images/the length of days in time-series
def trainModel(feature_sample0_train,feature_sample1_train,feature_sample0_val,feature_sample1_val, data_dim,timesteps):
    train_0_num=feature_sample0_train.shape[1]
    train_1_num=feature_sample1_train.shape[1]
    valid_num=feature_sample0_val.shape[1]
    data_train_0 = feature_sample0_train.T
    data_train_1 = feature_sample1_train.T
    train_data = np.concatenate((data_train_0,data_train_1))
    test_data = np.concatenate((feature_sample0_val.T,feature_sample1_val.T))
    train_num,dim=data_train_0.shape
    label_0 = np.zeros((train_num,1))
    label_1 = np.ones((train_num,1))
    train_label = np.concatenate((label_0,label_1))
    label_2 = np.zeros((valid_num,1))
    label_3 = np.ones((valid_num,1)) 
    test_label = np.concatenate((label_2,label_3))
    epochs = 100

    size_train = train_num*2
    size_test=valid_num*2
    train_data1 = train_data.reshape((size_train,data_dim,timesteps))
    test_data1 = test_data.reshape((size_test,data_dim,timesteps))
    train_data1 = np.swapaxes(train_data1,1,2)
    test_data1 = np.swapaxes(test_data1,1,2)
    print(train_data1.shape)

    model = Sequential()
    model.add(LSTM(32, return_sequences=True))  
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(32, return_sequences=True)) 
    model.add(Dropout(0.4))
    model.add(LSTM(32)) 
    model.add(Dense(1,activation='sigmoid'))

    Adam = optimizers.Adam(learning_rate=0.00)
    model.compile(optimizer = Adam,
          loss=[binary_focal_loss(alpha=0.5, gamma=2)],
          metrics=['accuracy']) #categorical_accuracy

    # learning schedule callback
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    history = model.fit(train_data1,train_label,
              epochs = 50,
              batch_size = 96,
              validation_data = (test_data1,test_label),
              callbacks = callbacks_list,
              verbose = 1)

    score = model.evaluate(test_data1, test_label, batch_size=32)
    print(score)
    return model

# example    
feature_sample0_train = np.load('./sample0.npy')
feature_sample1_train = np.load('./sample1.npy')
feature_sample0_val = np.load(r'./validation_sample0.npy')
feature_sample1_val = np.load(r'./validation_sample1.npy')

model=trainModel(feature_sample0_train,feature_sample1_train,feature_sample0_val,feature_sample1_val,8,17)
