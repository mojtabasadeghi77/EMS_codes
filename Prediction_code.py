#%% Please Change the code for different models

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
with h5py.File(path+"IR_test.mat", 'r') as f:
    IR = np.array(f["IR_test"])[..., np.newaxis]


# Location
with h5py.File(path+"LC_test.mat", 'r') as f:
    LC = np.array(f["LC_test"]).transpose([0, 2, 3, 1])

with h5py.File(path+"PMW_test.mat", 'r') as f:
    PMW = np.array(f["PMW_test"])[..., np.newaxis]

IR1, PMW1, LC1 = IR[:4],PMW[:4],LC[:4]

inp_res = np.concatenate([IR1, LC1], axis=-1)
print('load mat')
#%%

inp = keras.layers.Input(shape=(3000, 9000, 3))

conv1 = keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu")(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(pool1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = keras.layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(pool2)
conv5 = keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(conv3)
upsam1 = keras.layers.UpSampling2D(size=(2, 2))(conv5)
combine1 = keras.layers.Concatenate()([conv2, upsam1])
conv6 = keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(combine1)
upsam2 = keras.layers.UpSampling2D(size=(2, 2))(conv6)
conv7 = keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu")(upsam2)
combine2 = keras.layers.Concatenate()([conv1, conv7])
conv8 = keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu")(combine2)
conv9 = keras.layers.Conv2D(16, kernel_size=5, padding="same", activation="relu")(conv8)
conv10 = keras.layers.Conv2D(8, kernel_size=5, padding="same", activation="relu")(conv9)
out = keras.layers.Conv2D(1, kernel_size=5, padding="same", activation="relu")(conv9)



#%%
# load weights into new model and prediction
model = keras.models.Model(inputs= inp, outputs=out) 
print(model.summary())
from keras.models import load_model
model.load_weights('model_weights.h5')
print("Loaded weights of model from disk")
model.compile(loss="mse", optimizer="adam")
prediction = model.predict(inp_res, batch_size=1, verbose=True)
#shape = (None,3000,9000,2)
res = dict()
res["predict"] = prediction
res["IRtest"] = IR1
res["PMWtest"] = PMW1
savemat(path+"Global_Prediction_1", res)
print("Finished")