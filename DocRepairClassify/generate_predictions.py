from keras import models
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.preprocessing import image
import os
import Image
import para
import csv
import generator
import numpy.random as rng
from keras import backend as K
import keras.metrics
import time
from keras.callbacks import TensorBoard
from keras.utils.generic_utils import get_custom_objects
from SimulateDegradation import Dilation
from Verify import *
import math

def top_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
keras.metrics.top_accuracy=top_accuracy

def generate_predictions(img_path,model_path,lineseg_path,charseg_path):
    net = models.load_model(model_path)
    doc_img = Image.open(img_path)
    doc_img = np.array(doc_img)

    lines=[]
    with open(lineseg_path, newline='') as linefile:
        reader = csv.reader(linefile, delimiter=',', quotechar='|')
        for row in reader:
            lines.append([int((row[0].split())[0]),int((row[0].split())[1])])

    output=np.zeros((len(lines)*40*4,doc_img.shape[1]),dtype=np.uint8)+255
    for idx,line in enumerate(lines):
        output[idx*40*4:idx*40*4+40,:]=doc_img[line[0]:line[1]+1,:]

    chars=[]
    with open(charseg_path, newline='') as charfile:
        reader = csv.reader(charfile, delimiter=',', quotechar='|')
        for row in reader:
            if len(row)==0:
                continue
            chars.append([int((row[0].split())[0]),int((row[0].split())[1]),int((row[0].split())[2]),int((row[0].split())[3])])

    npz_healthy = np.load(para.data_result_path+'/healthy.npz')
    healthy_labels=npz_healthy['labels']
    healthy_patches0=npz_healthy['patches']
    healthy_patches=np.expand_dims(healthy_patches0,-1)

    line_count=0
    col_count=0
    top_n=3
    for char in chars:
        while lines[line_count][0]<char[2]:
            line_count+=1
            col_count=0
        im20 = np.copy(doc_img[char[2]:char[3]+1,char[0]:char[1]+1])

        im2 = np.expand_dims(im20,-1)
        im2 = np.expand_dims(im2,0)
        y_pred0=(net.predict_on_batch(im2))[0]
        max_pred=np.argsort(-y_pred0)[:top_n]
    
        img=Image.fromarray(im20,'L')
        img.save(para.data_result_path+'/prediction_results/'+str(line_count+1)+'_'+str(col_count+1)+'.png')

        f = open(para.data_result_path+'/prediction_results/'+str(line_count+1)+'_'+str(col_count+1)+'.txt', 'w')

        voters=[]
        for idx,i in enumerate(max_pred):
            f.write(str(healthy_labels[i])+' '+str(y_pred0[i])+'\r\n')
            output[line_count*40*4+(idx+1)*40:line_count*40*4+(idx+2)*40,char[0]:char[1]+1]=healthy_patches0[max_pred[idx]]

        f.write('\r\n')
        for idx,pred in enumerate(y_pred0):
            f.write(str(healthy_labels[idx])+' '+str(pred)+'\r\n')
    
        img=Image.fromarray(healthy_patches0[max_pred[0]],'L')
        img.save(para.data_result_path+'/prediction_results/'+str(line_count+1)+'_'+str(col_count+1)+'_.png')
    
        col_count+=1
        f.close()

    img = Image.fromarray(output, 'L')
    img.save(para.data_result_path+'/prediction_results/predictions.png')
    K.clear_session()

if __name__ == '__main__':
    img_name='labschoolreport-0002-012-8.tiff'
    fold_name='hard'
    img_path=os.path.join(para.data_result_path+'/data/test',fold_name,img_name)
    model_path=para.data_result_path+'/models/checkpoint_reduced_units.h5'
    lineseg_path=os.path.join(para.data_result_path+'/data/test\lines','lines_'+img_name+'.txt')
    charseg_path=os.path.join(para.data_result_path+'/RoI_results',img_name+'.txt')
    generate_predictions(img_path,model_path,lineseg_path,charseg_path)
