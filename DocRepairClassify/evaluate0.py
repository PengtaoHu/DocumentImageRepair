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

def top_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
keras.metrics.top_accuracy=top_accuracy

img_name='labschoolreport-0002-012-8.tiff'
fold_name='hard'

#get_custom_objects().update({'top_accuracy': TopAccuracy})

model_name='checkpoint'
net = models.load_model(para.data_result_path+'/models/'+model_name+'.h5')

doc_img = Image.open(os.path.join(para.data_result_path+'/data/test',fold_name,img_name))
doc_img = np.array(doc_img)

lines=[]
with open(os.path.join(para.data_result_path+'/data/test\lines','lines_'+img_name+'.txt'), newline='') as linefile:
    reader = csv.reader(linefile, delimiter=',', quotechar='|')
    for row in reader:
        lines.append([int((row[0].split())[0]),int((row[0].split())[1])])

multi=4
output=np.zeros((len(lines)*40*multi,doc_img.shape[1]),dtype=np.uint8)
for idx,line in enumerate(lines):
    output[idx*40*multi:idx*40*multi+40,:]=doc_img[line[0]:line[1]+1,:]

chars=[]
with open(os.path.join(para.data_result_path+'/data/test\chars','chars_'+img_name+'.txt'), newline='') as linefile:
    reader = csv.reader(linefile, delimiter=',', quotechar='|')
    for row in reader:
        chars.append([int((row[0].split())[0]),int((row[0].split())[1]),int((row[0].split())[2]),int((row[0].split())[3])])

npz_healthy = np.load(para.data_result_path+'/healthy.npz')
healthy_labels=npz_healthy['labels']
healthy_patches0=npz_healthy['patches']
healthy_patches=np.expand_dims(healthy_patches0,-1)

line_count=0
col_count=0
for char in chars:
    while lines[line_count][0]<char[2]:
        line_count+=1
        col_count=0
    im20 = doc_img[char[2]:char[3]+1,char[0]:char[1]+1]
    im2 = np.expand_dims(im20,-1)
    im2 = np.expand_dims(im2,0)
    ims2= np.stack((im2,)*healthy_patches.shape[0], axis=0)
    y_pred0=(net.predict_on_batch(im2))[0]
    max_pred=np.argsort(-y_pred0)[:multi-1]
    
    img=Image.fromarray(im20,'L')
    img.save(para.data_result_path+'/evaluate_results/'+str(line_count+1)+'_'+str(col_count+1)+'.png')

    f = open(para.data_result_path+'/evaluate_results/'+str(line_count+1)+'_'+str(col_count+1)+'.txt', 'w')
    for i in max_pred:
        f.write(str(y_pred0[i])+' '+str(healthy_labels[i])+'\r\n')
    for i in range(len(y_pred0)):
        f.write(str(y_pred0[i])+' '+str(healthy_labels[i])+'\r\n')
    f.close()
    
    for idx,value in enumerate(max_pred):
        output[line_count*40*multi+40*(idx+1):line_count*40*multi+40*(idx+2),char[0]:char[1]+1]=healthy_patches0[value]
    col_count+=1

img = Image.fromarray(output, 'L')
img.save(para.data_result_path+'/repair_results/repaired.png')
K.clear_session()


