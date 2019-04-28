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

def evaluate_diff(img_path,output_path,lineseg_path,charseg_path):
    net = models.load_model(model_path)
    doc_img = Image.open(img_path)
    doc_img = np.array(doc_img)

    lines=[]
    with open(lineseg_path, newline='') as linefile:
        reader = csv.reader(linefile, delimiter=',', quotechar='|')
        for row in reader:
            lines.append([int((row[0].split())[0]),int((row[0].split())[1])])

    multi=4
    output=np.zeros((len(lines)*40*multi,doc_img.shape[1]),dtype=np.uint8)+255
    for idx,line in enumerate(lines):
        output[idx*40*multi:idx*40*multi+40,:]=doc_img[line[0]:line[1]+1,:]

    output_final=np.zeros((doc_img.shape[0],doc_img.shape[1]),dtype=np.uint8)+255

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

    healthy_patches_group=[]
    for im in healthy_patches0:
        group=[]
        group.append(im)
        group.append(Dilation(im))
        group.append(Dilation(Dilation(im)))
        healthy_patches_group.append(group)

    line_count=0
    col_count=0
    for char in chars:
        while lines[line_count][0]<char[2]:
            line_count+=1
            col_count=0
        im0 = np.copy(doc_img[char[2]:char[3]+1,char[0]:char[1]+1])

        scores=[]
        x_shifts=[]
        y_shifts=[]
        for im1 in healthy_patches:
            result=SearchAlign(im0,im1)
            scores.append(result[0])
            y_shifts.append(result[1])
            x_shifts.append(result[2])

        min_scores=np.argsort(scores)[:multi-1]
    
        img=Image.fromarray(im0,'L')
        img.save(para.data_result_path+'/verify_results/'+str(line_count+1)+'_'+str(col_count+1)+'.png')

        f = open(para.data_result_path+'/verify_results/'+str(line_count+1)+'_'+str(col_count+1)+'.txt', 'w')
        for i in min_scores:
            f.write(str(scores[i])+' '+str(healthy_labels[i])+' '+str(y_shifts[i])+' '+str(x_shifts[i])+'\r\n')
        f.write('\r\n')
        for i in range(len(scores)):
            f.write(str(scores[i])+' '+str(healthy_labels[i])+' '+str(y_shifts[i])+' '+str(x_shifts[i])+'\r\n')
        f.close()

        output_fianl[char[2]:char[3]+1,char[0]:char[1]+1]=healthy_patches[min_scores[0]]
    
        for idx,value in enumerate(min_scores):
            output[line_count*40*multi+40*(idx+1):line_count*40*multi+40*(idx+2),char[0]:char[1]+1]=healthy_patches[value]
        col_count+=1
        break
    img=Image.fromarray(output_final,'L')
    img.save(output_path)
    img = Image.fromarray(output, 'L')
    img.save(para.data_result_path+'/repair_results/repair_visualization_diff.png')
    K.clear_session()

if __name__ == '__main__':
    img_name='labschoolreport-0002-012-8.tiff'
    fold_name='hard'
    img_path=os.path.join(para.data_result_path+'/data/test',fold_name,img_name)
    lineseg_path=os.path.join(para.data_result_path+'/data/test\lines','lines_'+img_name+'.txt')
    charseg_path=os.path.join(para.data_result_path+'/RoI_results',img_name+'.txt')
    output_path=para.data_result_path+'/repair_results/repair_output_diff.png'
    evaluate_vote(img_path,output_path,lineseg_path,charseg_path)
