#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 as cv
from sklearn.model_selection import LeaveOneOut
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from keras.layers import Dense
import tensorflow as tf
import random
from tensorflow.keras.optimizers import SGD


# In[2]:


DATADIR="C:/Users/razeeee/English Alphabet"
path=os.path.join(DATADIR)
classes=27
new_img=[]
label=[]
for i in range (1,27):
    for img in os.listdir(f'English Alphabet/{i}'):
        img_array=cv.imread(os.path.join(f'English Alphabet/{i}',img),cv.IMREAD_GRAYSCALE)
        img_array=img_array.reshape(-1)
        #image=[int(i) for i in image]
        if img_array is not None:
            new_img.append(img_array)
        label.append(i) 
y=np.array(label)  
x=np.array(new_img) 
label = to_categorical(y,num_classes=classes)
labels=label[ : ,1: ]
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        if labels[i,j]==0:
            labels[i,j]=-1


# In[3]:


def model(x_train,y_train,x_test,y_test):
    model=Sequential()
    model.add(Dense(26,input_shape=(3600, ),activation='softmax', kernel_initializer='zero'))
    model.compile(loss='categorical_crossentropy' ,optimizer=SGD(learning_rate=0.21), metrics=['accuracy'])
    model.fit(x_train,y_train, epochs=100,verbose=0)
    eva=model.evaluate(x_test,y_test, verbose=0) 
    return(eva[1])


# In[4]:


def LOOCV(x,labels):
    loocv=LeaveOneOut()
    loocv.get_n_splits(x)
    acc_list=[]
    for i,j in loocv.split(x):
        x_train,x_test=x[i],x[j]
        y_train,y_test=labels[i],labels[j]
        acc=model(x_train,y_train,x_test,y_test)
        acc_list.append(acc)
    acc_array=np.array(acc_list)
    #print(acc_list)
    return(acc_array.mean())


# In[5]:


def noisy(k,x,labels):
    samples=x.copy()
    for i in range(samples.shape[0]):
        black_list=[]
        for j in range(samples.shape[1]):
            if int(samples[i,j])==0:
                black_list.append(j)     
    m=int(k*len(black_list))
    andis=random.sample(black_list,m)
    for i in range(samples.shape[0]):
        for t in (andis):
            samples[i,t]=255
    return samples      


# In[6]:


xtest_15=noisy(0.15,x,labels)
xtest_25=noisy(0.25,x,labels)
acc_15=model(x,labels,xtest_15,labels)
acc_25=model(x,labels,xtest_25,labels)
print('accuracy of data with 15% of noise = ',acc_15*100)
print('accuracy of data with 25% of noise = ',acc_25*100)


# In[7]:


acc_loocv=LOOCV(x,labels)


# In[8]:


print('accuracy of data in LOOCV model = ',acc_loocv*100)

