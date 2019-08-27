import numpy as np
from numpy import genfromtxt
from keras.layers import Input, Dense
from keras.models import Model
import scipy.io as sio
import h5py


def model_trained(train,test,nfeatures):

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


#autoencoder.fit(train, train,
#                epochs=10,
#                batch_size=100,
#                shuffle=True,
#                validation_data=(test, test))

  autoencoder.load_weights("./weights_U")

  ypred=autoencoder.predict(test)
  sio.savemat('reconstruct_u.mat', {'ypred':ypred,'truth':test})

  yencoded=encoder.predict(train)
  print('shape of encoded',np.shape(yencoded))
  sio.savemat('encoded_u.mat', {'encoding':yencoded})

