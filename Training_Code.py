 #%% Change a few number in the following code for different models 
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
with h5py.File(path+"LC_norm.mat", 'r') as f:
    loc = np.array(f["LC_norm"]).transpose([0, 2, 3, 1])

IRtrain, PMWtrain, LCtrain = ir[:106000], pmw[:106000], loc[:106000]
IRtest, PMWtest, LCtest= ir[106000:], pmw[106000:], loc[106000:]
## Normalize data here.

## Concatenate Location data with IR
inp_data = np.concatenate([IRtrain, LCtrain], axis=-1)
out_data = PMWtrain
pmw = None

# Keras Model
inp = keras.layers.Input(shape=(64, 64, 3))

conv1 = keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu")(inp)
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(pool1)
pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = keras.layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(pool2)
conv4 = keras.layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(conv3)
conv5 = keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(conv4)
upsam1 = keras.layers.UpSampling2D(size=(2, 2))(conv5)
combine1 = keras.layers.Concatenate()([conv2, upsam1])
conv6 = keras.layers.Conv2D(64, kernel_size=5, padding="same", activation="relu")(combine1)
upsam2 = keras.layers.UpSampling2D(size=(2, 2))(conv6)
combine2 = keras.layers.Concatenate()([conv1, upsam2])
conv7 = keras.layers.Conv2D(32, kernel_size=5, padding="same", activation="relu")(combine2)
out = keras.layers.Conv2D(1, kernel_size=5, padding="same", activation="relu")(conv7)


# summarize layers
model = keras.models.Model(inputs= inp, outputs=out) 
% If droupout: model.add(Dropout(0.2))
print(model.summary())
# plot graph
#plot_model(model, to_file='multiple_inputs.png')

# Compile the model
model.compile(loss="mse", optimizer="adam")


# Train model
model.fit(inp_data, out_data, epochs=30, batch_size=64, verbose=True) # ,validation_data=(IRtest,STtest))

#%%
#Test model 
inp_res = np.concatenate([IRtest, LCtest], axis=-1)
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

