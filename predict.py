import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras import applications
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.models import load_model

# 读取h5文件
f = h5py.File('E:\\wildRecog\\ndarray_test.h5','r')
x_test = f['x'][:]  # f.keys()可以查看所有的主键
f.close()

model = load_model('VGG16-transferlearning.model')

for i in range(1531):
    
    preds = model.predict(x_test,verbose=0)
    print(np.max(preds))
