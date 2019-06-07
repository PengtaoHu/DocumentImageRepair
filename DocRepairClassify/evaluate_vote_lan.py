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

def possible_words(word):
    idx=[0]*len(word)
    words_list=[]
    while idx[0]<len(word[0]):
        word0=''
        for k,voters in enumerate(word):
            word0+=para.char_dict[voters[idx[k]][0]]
        idx[len(word)-1]+=1
        for i in range(len(word)-1,0,-1):
            if idx[i] == len(word[i]):
                idx[i]=0
                idx[i-1]+=1
            else:
                break
        words_list.append(word0)
    return words_list


def evaluate_vote(img_path,output_path,model_path,lineseg_path,charseg_path,segimg_path):
    net = models.load_model(model_path)
    doc_img = Image.open(img_path)
    doc_img = np.array(doc_img)

    RoI_img = Image.open(segimg_path)
    RoI_img = np.array(RoI_img)

    lines=[]
    with open(lineseg_path, newline='') as linefile:
        reader = csv.reader(linefile, delimiter=',', quotechar='|')
        for row in reader:
            lines.append([int((row[0].split())[0]),int((row[0].split())[1])])

    multi=4
    output=np.zeros((len(lines)*40*multi,doc_img.shape[1],3),dtype=np.uint8)+255
    for idx,line in enumerate(lines):
        output[idx*40*multi:idx*40*multi+40,:]=np.stack((doc_img[line[0]:line[1]+1,:],)*3, axis=-1)
        output[idx*40*multi+40*1:idx*40*multi+40*2,:]=np.stack((RoI_img[idx*40*4+40*3:idx*40*4+40*4,:],)*3, axis=-1)
        output[idx*40*multi+40*2:idx*40*multi+40*3,:]=np.stack((doc_img[line[0]:line[1]+1,:],)*3, axis=-1)
        output[idx*40*multi+40*3:idx*40*multi+40*4,:]=np.stack((doc_img[line[0]:line[1]+1,:],)*3, axis=-1)

    output_fianl=doc_img.copy()

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
    max_voters=3
    word=[]
    words_pos=[]
    word_count=[]

    for idx_c,char in enumerate(chars):

        if idx_c>0 and abs(char[0]-chars[idx_c-1][0])>35:
            #for k in range(1,len(word)-1):

            print(possible_words(word))
            for voters0,char0,count0 in zip(word,words_pos,word_count):
                repair_patch=np.copy(doc_img[char0[2]:char0[3]+1,char0[0]:char0[1]+1])
                if len(voters0)<=1:
                    if len(voters0)>0 and (voters0[0][1]>0.9 and voters0[0][2]<1.5 or voters0[0][1]>0.8 and voters0[0][2]<0.4):
                        original_patch=healthy_patches0[np.argwhere(healthy_labels==voters0[0][0])][0][0]
                        repair_patch=Shift(original_patch,voters0[0][3],voters0[0][4])
                else:
                    voters_patches=[]
                    for voter in voters0:
                        original_patch=healthy_patches0[np.argwhere(healthy_labels==voter[0])][0][0]
                        voters_patches.append(Shift(original_patch,voter[3],voter[4]))
                    for y in range(40):
                        for x in range(19):
                            flag=1
                            for i in range(1,len(voters0)):
                                if abs(int(voters_patches[i][y][x])-int(voters_patches[0][y][x]))>100:
                                    flag=0
                                    break
                            if flag==1:
                                repair_patch[y][x]=voters_patches[0][y][x]
    
                img=Image.fromarray(repair_patch,'L')
                img.save(para.data_result_path+'/evaluate_results_voter/'+str(count0[0])+'_'+str(count0[1])+'_.png')
                repaired_patch=Repair(repair_patch,im20)
                output_fianl[char0[2]:char0[3]+1,char0[0]:char0[1]+1]=repair_patch
                output[line_count*40*multi+40*2:line_count*40*multi+40*3,char0[0]:char0[1]+1]=np.stack((repair_patch,)*3, axis=-1)
                output[line_count*40*multi+40*3:line_count*40*multi+40*4,char0[0]:char0[1]+1]=repaired_patch
            word=[]
            words_pos=[]
            word_count=[]

        while lines[line_count][0]<char[2]:
            line_count+=1
            col_count=0
        im20 = np.copy(doc_img[char[2]:char[3]+1,char[0]:char[1]+1])

        im2 = np.expand_dims(im20,-1)
        im2 = np.expand_dims(im2,0)
        y_pred0=(net.predict_on_batch(im2))[0]
        max_pred=np.argsort(-y_pred0)[:max_voters]
    
        img=Image.fromarray(im20,'L')
        img.save(para.data_result_path+'/evaluate_results_voter/'+str(line_count+1)+'_'+str(col_count+1)+'.png')

        f = open(para.data_result_path+'/evaluate_results_voter/'+str(line_count+1)+'_'+str(col_count+1)+'.txt', 'w')

        voters=[]
        for i in max_pred:
            if y_pred0[i]>0.01 and healthy_labels[i] != 'and':
                score_min=float('inf')
                for idx,im in enumerate(healthy_patches_group[i]):
                    align_result=SearchAlign(im20,im)
                    if align_result[0]<score_min:
                        result_min=align_result
                voters.append((healthy_labels[i],y_pred0[i])+result_min+(idx,))
                f.write(str(voters[-1])+'\r\n')
        word.append(voters)
        words_pos.append(char)
        word_count.append((line_count+1,col_count+1))
    
        col_count+=1
        f.close()
        #if line_count>0:
        #    break

    img=Image.fromarray(output_fianl,'L')
    img.save(output_path)
    img = Image.fromarray(output, 'RGB')
    img.save(para.data_result_path+'/repair_results/repair_visualization.png')
    K.clear_session()

if __name__ == '__main__':
    img_name='labschoolreport-0002-012-8.tiff'
    fold_name='hard'
    img_path=os.path.join(para.data_result_path+'/data/test',fold_name,img_name)
    model_path=para.data_result_path+'/models/checkpoint_reduced_units.h5'
    lineseg_path=os.path.join(para.data_result_path+'/data/test\lines','lines_'+img_name+'.txt')
    charseg_path=os.path.join(para.data_result_path+'/RoI_results',img_name+'.txt')
    output_path=para.data_result_path+'/repair_results/repair_output.png'
    segimg_path=os.path.join(para.data_result_path+'/RoI_results',img_name+'.png')
    evaluate_vote(img_path,output_path,model_path,lineseg_path,charseg_path,segimg_path)
