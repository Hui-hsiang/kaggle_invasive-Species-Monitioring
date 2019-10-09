import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt


trainpath = str('train\\')
testpath = str('test\\')
n_tr = len(os.listdir(trainpath))
print('num of traing files:',n_tr)

train_labels = pd.read_csv('train_labels.csv')
train_labels.head()

from skimage import io, transform

sample_image = io.imread(trainpath + '1.jpg')
print('Height:{0} Width:{1}'.format(*sample_image.shape))
plt.imshow(sample_image)

x = np.empty(shape=(n_tr, 224, 224, 3))
y = np.empty(n_tr)
labels = train_labels.invasive.values

for k,v in enumerate(np.random.permutation(n_tr)):
    path = '{0}{1}.jpg'.format(trainpath, v+1)
    tr_im = io.imread(path)
    x[k] = transform.resize(tr_im, output_shape=(224, 224, 3))
    y[k] = labels[v]

x = x.astype('float32')  # elements in x are between 0 and 1 inclusively.

f = h5py.File('ndarray_train.h5','w')
f['x']=x
f['y']=y
f.close()

f = h5py.File('ndarray_train.h5','r')
x = f['x'][:]  # f.keys()可以查看所有的主键
y = f['y'][:]
f.close()

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
print(x_train.shape,y_train.shape,x_val.shape,y_val.shape, sep='\n')


from keras.models import Sequential, Model
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD


img_shape = (224, 224, 3)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=img_shape)
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

for layer in model.layers[:-1]:
    layer.trainable = False

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

batch_size = 32
epochs = 50

train_datagen = ImageDataGenerator(rotation_range=30, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True)

train_datagen.fit(x_train)


train_datagenerator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

history = model.fit_generator(
    train_datagenerator,
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    validation_steps=50,
    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)])

from skimage import io, transform

n_test = len(os.listdir(testpath))
xx = np.empty(shape=(n_test, 224, 224, 3))
xx = xx.astype('float32')

for i in range(n_test):
    path = '{0}{1}.jpg'.format(testpath, i+1)
    test_im = io.imread(path)
    xx[i] = transform.resize(test_im, output_shape=(224, 224, 3))


f = h5py.File('ndarray_test.h5','w')
f['x']=xx
f.close()


f = h5py.File('ndarray_test.h5','r')
x_test = f['x'][:]  # f.keys()可以查看所有的主键
f.close()


result = model.predict(x_test)

result[result>0.5] = 1
result[result!=1] = 0
result[0]

df = pd.read_csv('sample_submission.csv')
df.invasive = result.flatten()
df.head()

df.to_csv('demo_submission.csv', index=False)