 #%%
# qrsh -q gpu/gpu2

import os
import numpy as np
import random
random.seed(9001)
os.environ["KERAS_BACKEND"] = "theano"
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, floatX=float32"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda0, floatx=float32"
import keras
keras.backend.set_image_data_format('channels_last')
from scipy.io import loadmat, savemat
import h5py



#from matplotlib import pylab as plt
path = ""

# IR
with h5py.File(path+"IR_norm.mat", 'r') as f:
    ir = np.array(f["IR_norm"])[..., np.newaxis]

# PMW
with h5py.File(path+"PMW_norm.mat", 'r') as f:
    pmw = np.array(f["PMW_norm"])[..., np.newaxis]

# Location

IRtrain, PMWtrain = ir[:28000], pmw[:28000], 
IRtest, PMWtest= ir[28000:], pmw[28000:], 


## Concatenate Location data with IR
inp_data = IRtrain
out_data = PMWtrain
pmw = None

# Keras Model
inp = keras.layers.Input(shape=(128, 128, 1))

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
model = keras.models.Model(inputs= inp, outputs=out) 
print(model.summary())
# plot graph
#plot_model(model, to_file='multiple_inputs.png')

# Compile the model

model.compile(loss='mean_squared_error', optimizer='Adam')

% If droupout: model.add(Dropout(0.2))

# Train model
model.fit(inp_data, out_data, epochs=30, batch_size=32, verbose=True) # ,validation_data=(IRtest,STtest))

#%%
#Test model 
inp_res = IRtest
print('a')
out_res = PMWtest
print('b')
prediction = model.predict(inp_res, batch_size=1, verbose=True)
print('c')
res = dict()
res["predict"] = prediction
res["Inputs"] = inp_res
res["Outputs"] =  out_res
print('d')
savemat(path+"results", res)
#%%
weights_list = model.get_weights()
print(weights_list)
# Save the weights
model.save_weights('model_weights.h5')

weights_list = model.get_weights()
print(weights_list)

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

#%%

