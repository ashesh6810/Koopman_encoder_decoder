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

autoencoder.save_weights('./weights_U')


autoencoder.fit(train, train,
                epochs=10,
                batch_size=100,
                shuffle=True,
                validation_data=(test, test))

autoencoder.save_weights("./weights_U")

ypred=autoencoder.predict(test)
sio.savemat('reconstruct_u.mat', {'ypred':ypred,'truth':test})

yencoded=encoder.predict(train)
print('shape of encoded',np.shape(yencoded))
sio.savemat('encoded_u.mat', {'encoding':yencoded})

#ydecoded=decoder.predict(yencoded)

#print('shape of decoded',np.shape(ydecoded))
