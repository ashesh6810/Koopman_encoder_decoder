import numpy as np
from numpy import genfromtxt
from keras.layers import Input, Dense
from keras.models import Model
import scipy.io as sio
import h5py
from saved_model import model_trained


model_params = {'tau': 0.25,
                'nstep': 1000,
                'N': 11,
                'd': 22}
res_params = {'radius': 0.5,
              'degree': 3,
              'sigma': 0.5,
              'train_length': 80000,
              'num_inputs': model_params['N'],
              'predict_length': 2000,
              'beta': 0.001
              }
##
spin_off=0
shift_k=0
nfeatures=4900

file=h5py.File('cavity_2d.mat', 'r')
input_data=(np.array(file['U']))
data=(input_data)
M=np.mean(data.flatten())
sdev=np.std(data.flatten())
data=(data-M)/sdev
train = data[0:res_params['train_length'],:]
test = data[res_params['train_length']:res_params['train_length']+res_params['predict_length'],:]
print('np.shape(train)', np.shape(train))
print('np.shape(test)', np.shape(test))

##
trainN=80000
testN=2000
##


print('np.shape(train)', np.shape(train))
print('np.shape(test)', np.shape(test))

model_trained(train, test, nfeatures)
