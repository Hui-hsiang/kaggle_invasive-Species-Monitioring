import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt


trainpath = str('E:\\wildRecog\\train\\')
testpath = str('E:\\wildRecog\\test\\')

f = h5py.File('E:\\wildRecog\\ndarray_train.h5','r')
x = f['x'][:]  
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
print (model.summary())
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

batch_size = 32
epochs = 2

train_datagen = ImageDataGenerator(featurewise_center=True, rotation_range=30, zoom_range=0.2, width_shift_range=0.1,
                                   height_shift_range=0.1, horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(featurewise_center=True)

train_datagen.fit(x_train)
val_datagen.fit(x_val)

train_datagenerator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

history = model.fit_generator(
    train_datagenerator,
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)])

from skimage import io, transform

n_test = len(os.listdir(testpath))

# xx = np.empty(shape=(n_test, 224, 224, 3))
# xx = xx.astype('float32')

# for i in range(n_test):
#     path = '{0}{1}.jpg'.format(testpath, i+1)
#     test_im = io.imread(path)
#     xx[i] = transform.resize(test_im, output_shape=(224, 224, 3))


# f = h5py.File('E:\\wildRecog\\ndarray_test.h5','w')
# f['x']=xx
# f.close()


f = h5py.File('E:\\wildRecog\\ndarray_test.h5','r')
x_test = f['x'][:]  # f.keys()可以查看所有的主键
f.close()
test_generator = val_datagen.flow(x_test, batch_size=1, shuffle=False)

result = model.predict_generator(test_generator, n_test)

result[result>0.5] = 1
result[result!=1] = 0


df = pd.read_csv('E:\\wildRecog\\sample_submission.csv')
df.invasive = result.flatten()
df.head()

df.to_csv('E:\\wildRecog\\demo_submission.csv', index=False)