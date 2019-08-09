import numpy as np
from numpy import genfromtxt
from keras.layers import Input, Dense
from keras.models import Model
import scipy.io as sio
import h5py

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
##

## Read Data
file=h5py.File('encoded_u.mat', 'r')
input_data=(np.array(file['encoded']))
data=(input_data)
M=np.mean(data.flatten())
sdev=np.std(data.flatten())
data=(data-M)/sdev
train = data[0:res_params['train_length'],:]

print('np.shape(train)', np.shape(train))



encoding_dim =100  # this is our input placeholder
input_img = Input(shape=(nfeatures,))
# "encoded" is the encoded representation of the input
encoded = Dense(2000, activation='relu')(input_img)
encoded=Dense(1000,activation='relu')(encoded)
encoded=Dense(500,activation='relu')(encoded)
encoded=Dense(200,activation='relu')(encoded)
encoded=Dense(encoding_dim,activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded=Dense(200,activation='relu')(encoded)
decoded=Dense(500,activation='relu')(decoded)
decoded = Dense(1000, activation='relu')(decoded)
decoded = Dense(2000, activation='relu')(decoded)
decoded = Dense(nfeatures, activation=None)(decoded)


# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
encoded_input = Input(shape=(encoding_dim,))
deco=autoencoder.layers[-5](encoded_input)
deco=autoencoder.layers[-4](deco)
deco=autoencoder.layers[-3](deco)
deco=autoencoder.layers[-2](deco)
deco=autoencoder.layers[-1](deco)
decoder= Model(encoded_input, deco)

# retrieve the last layer of the autoencoder model
autoencoder.compile(optimizer='adadelta', loss='mse')
autoencoder.load_weights("./weights_U")

ydecoded=decoder.predict(train)

print('shape of decoded',np.shape(ydecoded))


sio.savemat('timevolution_u.mat', {'ydecoded_time':ydecoded})

