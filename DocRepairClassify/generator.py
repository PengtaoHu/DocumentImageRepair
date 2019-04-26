import para
import math
import numpy as np
import keras
from random import randint
import os
from keras.preprocessing import image
import Image
import csv
import scipy.misc
from SimulateDegradation import *
def DataGenerator(batch_size=para.batch_size,fold=0):
    npz_healthy = np.load(para.data_result_path+'/healthy.npz')
    healthy_labels=npz_healthy['labels']
    healthy_patches=npz_healthy['patches']
    while True:
        labels = []
        ims = []
        for i in range(batch_size):
            
            idx1=randint(0,len(healthy_patches)-1)
            idx2=randint(0,len(healthy_patches)-1)
            idx3=randint(0,len(healthy_patches)-1)
            img=HalfHalfPatch(healthy_patches[idx1],healthy_patches[idx2],healthy_patches[idx3],randint(-3,3))[0]
            img=np.expand_dims(img,-1)
            
            labels.append(idx2)
            ims.append(img)
        ims=np.array(ims)
        labels=keras.utils.to_categorical(labels,para.n_class)
        labels=np.array(labels)
        yield ims, labels

def ValidationDataGenerator(batch_size=para.batch_size,fold=0):
    npz_healthy = np.load(para.data_result_path+'/healthy.npz')
    healthy_labels=npz_healthy['labels']
    npz_train = np.load(para.data_result_path+'/train.npz')
    train_labels=npz_train['labels']
    train_patches=npz_train['patches']
    idx1=0
    while True:
        labels = []
        ims = []
        if idx1+batch_size-1>=train_labels.shape[0]:
            idx1=0
        for i in range(batch_size):
            img=np.expand_dims(train_patches[idx1],-1)
            label=(np.argwhere(healthy_labels==train_labels[idx1]))[0]
            idx1+=1
            labels.append(label)
            ims.append(img)
        ims=np.array(ims)
        labels=keras.utils.to_categorical(labels,para.n_class)
        labels=np.array(labels)
        yield ims, labels


def RoIGenerator(batch_size=para.batch_size,fold=0):
    npz_healthy = np.load(para.data_result_path+'/healthy.npz')
    healthy_labels=npz_healthy['labels']
    healthy_patches=npz_healthy['patches']
    while True:
        labels = []
        ims = []
        for i in range(batch_size):
            label=randint(0,1)
            idx1=randint(0,len(healthy_patches)-1)
            idx2=randint(0,len(healthy_patches)-1)
            idx3=randint(0,len(healthy_patches)-1)
            if label==1:
                if healthy_labels[idx2]=='space':
                    label=0
                img=HalfHalfPatch(healthy_patches[idx1],healthy_patches[idx2],healthy_patches[idx3],view_size=para.patch_size[1]*2)[0]
            else:
                shift_from_center=randint(-9,9)
                while shift_from_center==0:
                    shift_from_center=randint(-9,9)
                img=HalfHalfPatch(healthy_patches[idx1],healthy_patches[idx2],healthy_patches[idx3],shift_from_center,view_size=para.patch_size[1]*2)[0]
            img=np.expand_dims(img,-1)
            labels.append(label)
            ims.append(img)
        ims=np.array(ims)
        labels=keras.utils.to_categorical(labels,2)
        labels=np.array(labels)
        yield ims, labels
'''
def SegBarGenerator(batch_size=para.batch_size,fold=0):
    npz_healthy = np.load(para.data_result_path+'/healthy.npz')
    healthy_labels=npz_healthy['labels']
    healthy_patches=npz_healthy['patches']
    while True:
        labels = []
        ims = []
        for i in range(batch_size):
            label=randint(0,1)
            idx1=randint(0,len(healthy_patches)-1)
            idx2=randint(0,len(healthy_patches)-1)
            if label==1:
                img=SegBarPatch(healthy_patches[idx1],healthy_patches[idx2])[0]
            else:
                shift_from_center=randint(-19,19)
                while abs(shift_from_center)<2:
                    shift_from_center=randint(-19,19)
                img=SegBarPatch(healthy_patches[idx1],healthy_patches[idx2],shift_from_center)[0]
            img=np.expand_dims(img,-1)
            labels.append(label)
            ims.append(img)
        ims=np.array(ims)
        labels=keras.utils.to_categorical(labels,2)
        labels=np.array(labels)
        yield ims, labels
'''