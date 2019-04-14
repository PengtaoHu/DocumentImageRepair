from keras import models
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
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
from SimulateDegradation import Degrade,Shift
from random import randint
import para
import Image
import numpy as np
import cv2 as cv
from Verify import *

img_name='labschoolreport-0002-012-8.tiff'
fold_name='hard'

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
healthy_patches=npz_healthy['patches']

line_count=0
col_count=0
for char in chars:
    while lines[line_count][0]<char[2]:
        line_count+=1
        col_count=0
    im0 = doc_img[char[2]:char[3]+1,char[0]:char[1]+1]

    scores=[]
    x_shifts=[]
    y_shifts=[]
    for im1 in healthy_patches:
        result=SearchAlign(im0,im1)
        scores.append(result[0])
        y_shifts.append(result[1])
        x_shifts.append(result[2])

    min_scores=np.argsort(scores)[:3]
    
    img=Image.fromarray(im0,'L')
    img.save(para.data_result_path+'/verify_results0/'+str(line_count+1)+'_'+str(col_count+1)+'.png')

    f = open(para.data_result_path+'/verify_results0/'+str(line_count+1)+'_'+str(col_count+1)+'.txt', 'w')
    for i in min_scores:
        f.write(str(scores[i])+' '+str(healthy_labels[i])+' '+str(y_shifts[i])+' '+str(x_shifts[i])+'\r\n')
    f.write('\r\n')
    for i in range(len(scores)):
        f.write(str(scores[i])+' '+str(healthy_labels[i])+' '+str(y_shifts[i])+' '+str(x_shifts[i])+'\r\n')
    f.close()
    
    for idx,value in enumerate(min_scores):
        output[line_count*40*multi+40*(idx+1):line_count*40*multi+40*(idx+2),char[0]:char[1]+1]=healthy_patches[value]
    col_count+=1

img = Image.fromarray(output, 'L')
img.save(para.data_result_path+'/repair_results/repaired.png')
K.clear_session()



