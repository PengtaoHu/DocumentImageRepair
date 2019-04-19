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

def Repair(im0,im):
    output=np.zeros((40,19,3),dtype=np.uint8)
    for i in range(40):
        for j in range(19):
            if abs(im0[i][j].astype(int)-im[i][j].astype(int))<50:
                output[i][j][0]=im0[i][j]
                output[i][j][1]=im0[i][j]
                output[i][j][2]=im0[i][j]
            elif im0[i][j]>im[i][j]:
                output[i][j][0]=255
                output[i][j][1]=173
                output[i][j][2]=173
            else:
                output[i][j][0]=0
                output[i][j][1]=104
                output[i][j][2]=6
    return output

def top_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
keras.metrics.top_accuracy=top_accuracy

img_name='labschoolreport-0002-012-8.tiff'
fold_name='hard'

#get_custom_objects().update({'top_accuracy': TopAccuracy})

model_name='checkpoint_reduced_units'
net = models.load_model(para.data_result_path+'/models/'+model_name+'.h5')

doc_img = Image.open(os.path.join(para.data_result_path+'/data/test',fold_name,img_name))
doc_img = np.array(doc_img)

lines=[]
with open(os.path.join(para.data_result_path+'/data/test\lines','lines_'+img_name+'.txt'), newline='') as linefile:
    reader = csv.reader(linefile, delimiter=',', quotechar='|')
    for row in reader:
        lines.append([int((row[0].split())[0]),int((row[0].split())[1])])


output=np.zeros((len(lines)*40*3,doc_img.shape[1],3),dtype=np.uint8)+255
for idx,line in enumerate(lines):
    output[idx*40*3:idx*40*3+40,:]=np.stack((doc_img[line[0]:line[1]+1,:],)*3, axis=-1)

chars=[]
with open(os.path.join(para.data_result_path+'/data/test\chars','chars_'+img_name+'.txt'), newline='') as linefile:
    reader = csv.reader(linefile, delimiter=',', quotechar='|')
    for row in reader:
        chars.append([int((row[0].split())[0]),int((row[0].split())[1]),int((row[0].split())[2]),int((row[0].split())[3])])

npz_healthy = np.load(para.data_result_path+'/healthy.npz')
healthy_labels=npz_healthy['labels']
healthy_patches0=npz_healthy['patches']
healthy_patches=np.expand_dims(healthy_patches0,-1)

healthy_patches_group=[]
for im in healthy_patches0:
    group=[]
    group.append(im)
    group.append(Dilation(im))
    group.append(Dilation(Dilation(im)))
    healthy_patches_group.append(group)

line_count=0
col_count=0
max_voters=3
for idx,char in enumerate(chars):
    while lines[line_count][0]<char[2]:
        line_count+=1
        col_count=0
    im20 = np.copy(doc_img[char[2]:char[3]+1,char[0]:char[1]+1])

    im2 = np.expand_dims(im20,-1)
    im2 = np.expand_dims(im2,0)
    ims2= np.stack((im2,)*healthy_patches.shape[0], axis=0)
    y_pred0=(net.predict_on_batch(im2))[0]
    max_pred=np.argsort(-y_pred0)[:max_voters]
    
    img=Image.fromarray(im20,'L')
    img.save(para.data_result_path+'/evaluate_results_voter/'+str(line_count+1)+'_'+str(col_count+1)+'.png')

    f = open(para.data_result_path+'/evaluate_results_voter/'+str(line_count+1)+'_'+str(col_count+1)+'.txt', 'w')

    voters=[]
    for i in max_pred:
        if y_pred0[i]>0.01:
            score_min=float('inf')
            for idx,im in enumerate(healthy_patches_group[i]):
                align_result=SearchAlign(im20,im)
                if align_result[0]<score_min:
                    result_min=align_result
            voters.append((healthy_labels[i],y_pred0[i])+result_min+(idx,))
            f.write(str(voters[-1])+'\r\n')
    
    repair_patch=np.copy(doc_img[char[2]:char[3]+1,char[0]:char[1]+1])
    if len(voters)<2 or voters[0][2]<0.2:
        if len(voters)>0 and (voters[0][1]>0.9 and voters[0][2]<1.5 or voters[0][1]>0.8 and voters[0][2]<0.4 or voters[0][2]<0.2):
            original_patch=healthy_patches0[np.argwhere(healthy_labels==voters[0][0])][0][0]
            repair_patch=Shift(original_patch,voters[0][3],voters[0][4])
    else:
        voters_patches=[]
        for voter in voters:
            original_patch=healthy_patches0[np.argwhere(healthy_labels==voter[0])][0][0]
            voters_patches.append(Shift(original_patch,voter[3],voter[4]))

        for y in range(40):
            for x in range(19):
                flag=1
                for i in range(1,len(voters)):
                    if abs(int(voters_patches[i][y][x])-int(voters_patches[0][y][x]))>100:
                        flag=0
                        break
                if flag==1:
                    repair_patch[y][x]=voters_patches[0][y][x]
    
    img=Image.fromarray(repair_patch,'L')
    img.save(para.data_result_path+'/evaluate_results_voter/'+str(line_count+1)+'_'+str(col_count+1)+'_.png')
    repaired_patch=Repair(repair_patch,im20)
    output[line_count*40*3+40*1:line_count*40*3+40*2,char[0]:char[1]+1]=np.stack((repair_patch,)*3, axis=-1)
    output[line_count*40*3+40*2:line_count*40*3+40*3,char[0]:char[1]+1]=repaired_patch
    
    col_count+=1
    f.close()

img = Image.fromarray(output, 'RGB')
img.save(para.data_result_path+'/repair_results/real_repaired.png')
K.clear_session()


