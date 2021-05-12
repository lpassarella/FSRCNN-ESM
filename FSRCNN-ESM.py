import numpy as np

hr=np.load("hr.npy")
hr_flns=np.load("flns_120.npy")

hr_fsns=np.load("fsns_120.npy")

from netCDF4 import Dataset
v=Dataset("out2.nc")

lon = v.variables["lon"][:]
lat = v.variables["lat"][:]

lat_begin=np.where(lat == 0.125)


lat_end=np.where(lat == 60.125)

lon_begin =np.where(lon == 240.625)

lon_end= np.where(lon == 300.625)

lat_begin=int(lat_begin[0])

lat_end=int(lat_end[0])

lon_begin=int(lon_begin[0])

lon_end=int(lon_end[0])
hr_prec=v.variables["PRECC"][:]
#lr = np.reshape(lr,(360,720,1440))
#hr = np.reshape(hr,(360,720,1440))
hr=hr[:,lat_begin:lat_end,lon_begin:lon_end]
hr_fsns=hr_fsns[:,lat_begin:lat_end,lon_begin:lon_end]
hr_flns=hr_flns[:,lat_begin:lat_end,lon_begin:lon_end]
hr_prec=hr_prec[:,lat_begin:lat_end,lon_begin:lon_end]
lat=lat[lat_begin:lat_end]
lon=lon[lon_begin:lon_end]
print(hr.shape)
print(hr_prec.shape)
v=Dataset("out.nc")
hr_precl=v.variables["PRECL"][:]
hr_precl=hr_precl[:,lat_begin:lat_end,lon_begin:lon_end]
print(hr_precl.shape)
hr=np.reshape(hr,(360,240,240))
hr = np.concatenate((hr,hr_fsns,hr_flns),axis=0)
print(hr_precl.shape)
from sklearn.preprocessing import MinMaxScaler
hr = np.reshape(hr,(1080,240*240*1))
scaler = MinMaxScaler()
hr=scaler.fit_transform(hr)
hr = np.reshape(hr,(1080,240,240,1))

hr_precs=np.concatenate((hr_prec,hr_precl),axis=0)
hr_precs = np.reshape(hr_precs,(720,240*240*1))
hr_precs=scaler.fit_transform(hr_precs)
hr_precs = np.reshape(hr_precs,(720,240,240,1))

hr=np.concatenate((hr,hr_precs),axis=0)
print(hr.shape)

lr=np.zeros((1800,60,60))
import cv2
import numpy as np
for i in range(hr.shape[0]):
    lr[i] = cv2.resize(hr[i], dsize=(60, 60), interpolation=cv2.INTER_CUBIC)

res=np.zeros((1800,240,240))
import cv2
import numpy as np
for i in range(lr.shape[0]):
    res[i] = cv2.resize(lr[i], dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import Activation
res = np.reshape(res,(1800,240,240,1))
hr = np.reshape(hr,(1800,240,240,1))
lr = np.reshape(lr,(1800,60,60,1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(lr, hr, test_size=0.1, random_state=42)


from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, add, Conv2DTranspose
model = Sequential()
import keras
from keras.layers.advanced_activations import PReLU
act = keras.layers.advanced_activations.PReLU(init='zero', weights=None)
model.add(Conv2D(64, 9, padding='same',input_shape=(60,60,1)))
model.add(act)
model.add(Conv2D(32, 1, padding='same', kernel_initializer='he_normal'))
model.add(PReLU())
model.add(Conv2D(16, 3, padding='same', kernel_initializer='he_normal'))
model.add(PReLU())
model.add(Conv2D(12, 3, padding='same', kernel_initializer='he_normal'))
model.add(PReLU())
model.add(Conv2D(12, 3, padding='same', kernel_initializer='he_normal'))
model.add(PReLU())
model.add(Conv2DTranspose(1, 3, strides=(4, 4), padding='same'))


model.add(Conv2D(64, 3, padding='same'))
model.add(Conv2D(32, 3, padding='same'))
model.add(Conv2D(1, 1, padding='same'))

model.summary()

from timeit import default_timer as timer
import keras
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)
        
cb = TimingCallback()

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

history=model.fit(X_train ,y_train, validation_split=0.1,batch_size=10,epochs=100,shuffle=True,callbacks=[cb])


print(history.history.keys())
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
np.save('train_loss_fsrcnn_2.npy',train_loss)
np.save('val_loss_fsrcnn_2.npy',val_loss)
time=cb.logs
np.save('time_fsrcnn_2.npy',time)
np.save('X_test_fsrcnn.npy',X_test)
np.save('y_test_fsrcnn.npy',y_test)
model.save('my_model_fsrcnn_2.h5')