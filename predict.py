#%%

import os
import numpy as np
import random
random.seed(9001)
os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, floatx=float32"
import keras
keras.backend.set_image_data_format('channels_last')
# from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import Conv2DTranspose as Conv2DTran
import numpy as np


#%%
#load input data
from scipy.io import loadmat, savemat
path = ""
from scipy.io import loadmat, savemat
import h5py
#from matplotlib import pylab as plt
path = ""

# IR
with h5py.File(path+"IR_norm.mat", 'r') as f:
    IR = np.array(f["IR_norm"])[..., np.newaxis]



IR1 = IR

inp_res = IR1
print('load mat')
#%%


# Keras Model
inp = keras.layers.Input(shape=(1744, 1000, 1))

conv1 = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu")(inp)
conv11=keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(conv1)
conv111=keras.layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(conv11)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv11)
conv2 = keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(pool1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(pool2)
pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = keras.layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(pool3)
upsam1 = keras.layers.UpSampling2D(size=(2, 2))(conv3)
conv6 = keras.layers.Conv2D(256, kernel_size=5, padding="same", activation="relu")(upsam1)
conv66 = keras.layers.Conv2D(512, kernel_size=9, padding="same", activation="relu")(conv6)
upsam2 = keras.layers.UpSampling2D(size=(2, 2))(conv6)
conv8 = keras.layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(upsam2)
conv9 = keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(conv8)
conv10 = keras.layers.Conv2D(16, kernel_size=5, padding="same", activation="relu")(conv9)
out = keras.layers.Conv2D(1, kernel_size=5, padding="same", activation="relu")(conv10)

# summarize layers
# load weights into new model and prediction
model = keras.models.Model(inputs= inp, outputs=out) 
% If droupout: model.add(Dropout(0.2))
print(model.summary())
from keras.models import load_model
model.load_weights('model_weights.h5')
print("Loaded weights of model from disk")
model.compile(loss='mean_squared_error', optimizer='Adam')
prediction = model.predict(inp_res, batch_size=1, verbose=True)
#shape = (None,3000,9000,2)
res = dict()
res["predict"] = prediction
res["IRtest"] = IR1

savemat(path+"CONUS_Prediction", res)
print("Finished")