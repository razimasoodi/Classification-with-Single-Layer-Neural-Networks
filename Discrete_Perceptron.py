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
from sklearn.metrics import accuracy_score


# In[2]:


DATADIR="C:/Users/razeeee/English Alphabet"
path=os.path.join(DATADIR)
classes=27
new_img=[]
label=[]
for i in range (1,27):
    for img in os.listdir(f'English Alphabet/{i}'):
        img_array=cv.imread(os.path.join(f'English Alphabet/{i}',img),cv.IMREAD_GRAYSCALE)
        ret, image=cv.threshold(img_array,127,1,cv.THRESH_BINARY)
        image=image.reshape(-1)
        image=[int(i) for i in image]
        if image is not None:
            new_img.append(image)
        label.append(i) 
y=np.array(label)  
x=np.array(new_img) 
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if x[i,j]==0:
            x[i,j]=-1
label = to_categorical(y,num_classes=classes)
labels=label[ : ,1: ]
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        if labels[i,j]==0:
            labels[i,j]=-1


# In[3]:


def percep(x,labels):
    W=np.zeros((x.shape[1]+1,labels.shape[1]))   #n*m
    alpha=1
    x0=np.ones((x.shape[0],1))
    X=np.hstack((x0,x))
    y=np.zeros((1,labels.shape[1]))
    for k in range(10):
        for j in range(X.shape[0]):
            y=X[j].reshape((1,-1))@W
            ma=np.argmax(y)
            y[ : ]=-1
            y[ : ,ma]=1
            t=labels[j].reshape((1,-1))
            new=alpha*(t-y)
            #if t.all()!=y.all():
            W+=X[j].reshape((-1,1))@new
                    
    return W


# In[4]:


def LOOCV1(x,labels):
    loocv=LeaveOneOut()
    loocv.get_n_splits(x)
    acc_list=[]
    for i,j in loocv.split(x):
        #print('j=',j)
        x_train,x_test=x[i],x[j]
        y_train,y_test=labels[i],labels[j]
        x_test
        w=percep(x_train,y_train)
        x0=np.ones((x_test.shape[0],1))
        X=np.hstack((x0,x_test))
        pre=X@w
        for i in range(len(pre)):
            z=np.argmax(pre[i],axis=0)
            q=np.zeros((1,26))
            q[ : ]= -1
            q[ : ,z]= 1
            acc=accuracy_score(y_test,q)
            #print(acc)
        acc_list.append(acc)
    acc_array=np.array(acc_list)
    #print(acc_list)
    return(acc_list,acc_array.mean())


# In[5]:


#noisy
def noisy(k,x,labels):
    sample=x.copy()
    for i in range(sample.shape[0]):
        black_list=[]
        for j in range(sample.shape[1]):
            if sample[i,j]==-1:
                black_list.append(j)           
        m=int(k*len(black_list))
    for i in range(sample.shape[0]):
        for t in range(int(m)):
            a=(random.choice(black_list))
            sample[i,a]=1
    w=percep(x,labels)
    x0=np.ones((520,1))
    X=np.hstack((x0,sample))
    pre=X@w
    l=[]
    for i in range(len(pre)):
            z=np.argmax(pre[i],axis=0)
            q=np.zeros((1,26))
            q[ : ]= -1
            q[ : ,z]= 1
            l.append(q)
    l=np.array(l)
    l=l.reshape((520,26))
    acc=accuracy_score(labels,l)  
    return acc  


# In[6]:


a,b=LOOCV1(x,labels)
print('The accuracy of model is : ',b*100)


# In[7]:


acc_15=noisy(0.15,x,labels)
acc_25=noisy(0.25,x,labels)
print('accuracy of data with 15% of noise = ',acc_15*100)
print('accuracy of data with 25% of noise = ',acc_25*100)

