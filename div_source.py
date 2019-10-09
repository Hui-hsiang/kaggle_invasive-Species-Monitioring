import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt


trainpath = str('E:\\wildRecog\\train\\')
testpath = str('E:\\wildRecog\\test\\')
n_tr = len(os.listdir(trainpath))
print('num of traing files:',n_tr)

train_labels = pd.read_csv('E:\\wildRecog\\train_labels.csv')
train_labels.head()

from skimage import io, transform



x = np.empty(shape=(n_tr, 224, 224, 3))
y = np.empty(n_tr)
labels = train_labels.invasive.values

for k,v in enumerate(np.random.permutation(n_tr)):
    path = '{0}{1}.jpg'.format(trainpath, v+1)
    tr_im = io.imread(path)
    x[k] = transform.resize(tr_im, output_shape=(224, 224, 3))
    y[k] = labels[v]

x = x.astype('float32')  # elements in x are between 0 and 1 inclusively.

f = h5py.File('E:\\wildRecog\\ndarray_train.h5','w')
f['x']=x
f['y']=y
f.close()