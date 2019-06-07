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
from math import floor


def evaluate_RoI(img_path,output_path,model_path,lineseg_path):
    #generate RoI results
    #output: characters segmentation
    net = models.load_model(model_path)
    doc_img = Image.open(img_path)
    doc_img = np.array(doc_img)

    lines=[]
    with open(lineseg_path, newline='') as linefile:
        reader = csv.reader(linefile, delimiter=',', quotechar='|')
        for row in reader:
            lines.append([int((row[0].split())[0]),int((row[0].split())[1])])

    output=np.zeros((len(lines)*40*4,doc_img.shape[1]),dtype=np.uint8)
    for idx,line in enumerate(lines):
        output[idx*40*4:idx*40*4+40,:]=doc_img[line[0]:line[1]+1,:]

    npz_healthy = np.load(para.data_result_path+'/healthy.npz')
    healthy_labels=npz_healthy['labels']
    healthy_patches0=npz_healthy['patches']
    healthy_patches=np.expand_dims(healthy_patches0,-1)

    f_char=open(output_path+'.txt', 'w')
    for line_idx,line in enumerate(lines):
        predicts=[]
        f0 = open(output_path+str(line_idx+1)+'.txt', 'w')
        for i in range(doc_img.shape[1]-38):
            im20 = doc_img[line[0]:line[1]+1,i:i+38]
            im2 = np.expand_dims(im20,-1)
            im2 = np.expand_dims(im2,0)
            y_pred0=(net.predict_on_batch(im2))[0]
            predicts.append(y_pred0)
        predicts=np.asarray(predicts)
        predicts=predicts[:,1]

        for idx,value in enumerate(predicts):
            output[line_idx*40*4+40:line_idx*40*4+40*2,idx+19]=255*value

        suppressed_predicts=np.zeros_like(predicts)
        d=3
        for idx, value in enumerate(predicts):
            left=idx-d if idx>=d else 0
            right=idx+d+1 if idx+d<=predicts.shape[0] else predicts.shape[0]
            if value==np.amax(predicts[left:right]):
                suppressed_predicts[idx]=value
        #thres=0.25
        thres=0.4
        centers=[]
        for idx,value in enumerate(suppressed_predicts):
            if value>thres:
                f0.write(str(value)+'\r\n')
                centers.append(idx+19)
                output[line_idx*40*4+40*2:line_idx*40*4+40*3,idx+19]=255*value
          
        def merge(centers):
            flag=0
            centers_merged=[]
            centers_merged.append(centers[0])
            for idx in range(1,len(centers)-1):
                if centers[idx]-centers_merged[-1]<=12 and centers[idx]-centers_merged[-1]<centers[idx+1]-centers[idx]:
                    left=centers_merged.pop(-1)
                    centers_merged.append(floor((centers[idx]+left)/2))
                    flag=1
                else:
                    centers_merged.append(centers[idx])
            centers_merged.append(centers[-1])
            if centers_merged[-1]-centers_merged[-2]<=12:
                right=centers_merged.pop(-1)
                left=centers_merged.pop(-1)
                centers_merged.append(floor((left+right)/2))
                flag=1
            return centers_merged,flag

        flag=1
        while flag==1:
            centers_merged,flag=merge(centers)
            centers=centers_merged.copy()

        for value in centers_merged:
            f_char.write(str(value-9)+' '+str(value+9)+' '+str(line[0])+' '+str(line[1])+'\r\n')
            output[line_idx*40*4+40*3:line_idx*40*4+40*4,value]=255
        f0.close()
    f_char.close()
    img = Image.fromarray(output, 'L')
    img.save(output_path+'.png')
    K.clear_session()
    

if __name__ == '__main__':
    img_name='labschoolreport-0002-033-4.tiff'
    fold_name='hard'
    img_path=os.path.join(para.data_result_path+'/data/test',fold_name,img_name)
    model_path=para.data_result_path+'/models/checkpoint_RoI.h5'
    lineseg_path=os.path.join(para.data_result_path+'/data/test\lines','lines_'+img_name+'.txt')
    output_path=para.data_result_path+'/RoI_results/'+str(img_name)
    evaluate_RoI(img_path,output_path,model_path,lineseg_path)
