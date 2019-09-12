
import matplotlib
matplotlib.use('agg')

import os
import pickle
import numpy as np
import pandas as pd
#import xarray as xr
#import seaborn as sns
from keras.layers import Input, Convolution2D, Convolution1D, MaxPooling2D, Dense, Dropout, \
                          Flatten, concatenate, Activation, Reshape, \
                          UpSampling2D,ZeroPadding2D
from keras import layers

import keras
from pylab import plt
import tensorflow as tf
from matplotlib import cm
import scipy.io as sio
import h5py

Nlat=70 
Nlon=70 
trainN=80000
testN=1000
batch_size = 32
num_epochs = 8
pool_size = 2
drop_prob=0.0
conv_activation='relu'

n_channels=1



data_aug=np.zeros([100001,100,100])


f = h5py.File('Re30K_N70.mat','r')
data=f.get('u')
print('size of data:',np.shape(data))
data=np.array(data)
data_aug[:,15:85,15:85]=data[:,:,:].copy()
#file=sio.loadmat('Re30K_N70.mat')
#u=file['u']

MEAN_2D_U=np.mean(data_aug,axis=0)


m=np.mean(data_aug.flatten())
sdev=np.std(data_aug.flatten())

data_aug=(data_aug-m)/sdev

u=np.reshape(data_aug,(np.size(data_aug,0),100,100,1))
print('size of u:',np.shape(u))


x_train=u[0:trainN,:,:,:]
x_test=u[trainN:trainN+testN,:,:,:]

### print shape out####
print('size of training:',np.shape(x_train))
print('size of test',np.shape(x_test))

#MEAN_2D_U=np.mean(data_aug,axis=0)

print('time mean:', np.shape(MEAN_2D_U))


def build_model(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):

    model = keras.Sequential([
            
            ## Convolution with dimensionality reduction (similar to Encoder in an autoencoder)
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation, input_shape=(100,100,n_channels)),
            layers.MaxPooling2D(pool_size=pool_size),
            Dropout(drop_prob),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            layers.MaxPooling2D(pool_size=pool_size),
            # end "encoder"
            
            
            # dense layers (flattening and reshaping happens automatically)
            ] + [layers.Dense(hidden_size, activation='sigmoid') for i in range(n_hidden_layers)] +
             
            [
            
            
            # start "Decoder" (mirror of the encoder above)
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            layers.UpSampling2D(size=pool_size),
            Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
            layers.UpSampling2D(size=pool_size),
            layers.Convolution2D(n_channels, kernel_size, padding='same', activation=None)
            ]
            )
    
    
    optimizer= keras.optimizers.adam(lr=lr)

            
    model.compile(loss='mean_squared_error', optimizer = optimizer)
    
    return model

params = {'conv_depth': 32, 'hidden_size': 500,
              'kernel_size': 6, 'lr': 0.0001, 'n_hidden_layers': 0}

model = build_model(**params)
print(model.summary())
hist = model.fit(x_train, x_train,
                       batch_size = batch_size,
             verbose=1,shuffle=True,
             epochs = num_epochs,
             validation_data=(x_test,x_test),
             callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=5, # just to make sure we use a lot of patience before stopping
                                        verbose=0, mode='auto'),
                       keras.callbacks.ModelCheckpoint('best_weights.h5', monitor='val_loss',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode='auto', period=1)]
             )

print('finished training')
    # get best model from the training (based on validation loss),
    # this is neccessary because the early stopping callback saves the model "patience" epochs after the best one
model.load_weights('best_weights.h5')


pred=model.predict(x_test)

sio.savemat('prediction.mat', {'prediction':pred,'truth':x_test,'MEAN':m,'SDEV':sdev})
sio.savemat('Tmean.mat', {'time_mean':MEAN_2D_U})



