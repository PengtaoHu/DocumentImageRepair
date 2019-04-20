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

img_name='labschoolreport-0002-012-8.tiff'
fold_name='hard'

#get_custom_objects().update({'top_accuracy': TopAccuracy})

model_name='checkpoint_RoI'
net = models.load_model(para.data_result_path+'/models/'+model_name+'.h5')

doc_img = Image.open(os.path.join(para.data_result_path+'/data/test',fold_name,img_name))
doc_img = np.array(doc_img)

lines=[]
with open(os.path.join(para.data_result_path+'/data/test\lines','lines_'+img_name+'.txt'), newline='') as linefile:
    reader = csv.reader(linefile, delimiter=',', quotechar='|')
    for row in reader:
        lines.append([int((row[0].split())[0]),int((row[0].split())[1])])

output=np.zeros((len(lines)*40*2,doc_img.shape[1]),dtype=np.uint8)
for idx,line in enumerate(lines):
    output[idx*40*2:idx*40*2+40,:]=doc_img[line[0]:line[1]+1,:]

npz_healthy = np.load(para.data_result_path+'/healthy.npz')
healthy_labels=npz_healthy['labels']
healthy_patches0=npz_healthy['patches']
healthy_patches=np.expand_dims(healthy_patches0,-1)

for line_idx,line in enumerate(lines):
    predicts=[]
    f0 = open(para.data_result_path+'/evaluate_results/line'+str(line_idx)+'.txt', 'w')
    for i in range(doc_img.shape[1]-19):
        im20 = doc_img[line[0]:line[1]+1,i:i+19]
        im2 = np.expand_dims(im20,-1)
        im2 = np.expand_dims(im2,0)
        #ims2= np.stack((im2,)*healthy_patches.shape[0], axis=0)
        y_pred0=(net.predict_on_batch(im2))[0]
        predicts.append(y_pred0)
        f0.write(str(y_pred0)+'\r\n')
    f0.close()
    for idx,value in enumerate(predicts):
        output[line_idx*40*2+40:line_idx*40*2+40*2,idx+9]=255*value[1]
img = Image.fromarray(output, 'L')
img.save(para.data_result_path+'/repair_results/repaired.png')
