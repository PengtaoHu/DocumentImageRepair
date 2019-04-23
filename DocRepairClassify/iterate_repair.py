from evaluate_RoI import evaluate_RoI
from evaluate_vote import evaluate_vote
import os
import para
import numpy as np
import Image
import csv

img_name='labschoolreport-0002-012-8.tiff'
fold_name='hard'
img_path=os.path.join(para.data_result_path+'/data/test',fold_name,img_name)
model_path_RoI=para.data_result_path+'/models/checkpoint_RoI.h5'
lineseg_path=os.path.join(para.data_result_path+'/data/test\lines','lines_'+img_name+'.txt')
model_path_class=para.data_result_path+'/models/checkpoint_reduced_units.h5'
n_iteration=6
'''
for i in range(n_iteration):
    print(i)
    output_path=para.data_result_path+'/iterate/'+str(i)+'.txt'
    evaluate_RoI(img_path,output_path,model_path_RoI,lineseg_path)
    charseg_path=output_path
    output_path=para.data_result_path+'/iterate/'+str(i)+'.png'
    evaluate_vote(img_path,output_path,model_path_class,lineseg_path,charseg_path)
    img_path=output_path
'''
lines=[]
with open(lineseg_path, newline='') as linefile:
    reader = csv.reader(linefile, delimiter=',', quotechar='|')
    for row in reader:
        lines.append([int((row[0].split())[0]),int((row[0].split())[1])])

img_path=os.path.join(para.data_result_path+'/data/test',fold_name,img_name)
doc_img=Image.open(img_path)
doc_img=np.array(doc_img)
output=np.zeros((40*(n_iteration+1)*len(lines),doc_img.shape[1]),dtype=np.uint8)+255

for i in range(n_iteration+1):
    for idx,line in enumerate(lines):
        output[idx*40*(n_iteration+1)+40*i:idx*40*(n_iteration+1)+40*(i+1),:]=doc_img[line[0]:line[1]+1,:]
    if i<n_iteration:
        doc_img=Image.open(para.data_result_path+'/iterate/'+str(i)+'.png')
        doc_img=np.array(doc_img)

img = Image.fromarray(output, 'L')
img.save(para.data_result_path+'/iterate/multi.png')

'''
charseg_path=para.data_result_path+'/iterate/char.txt'
evaluate_RoI(img_path,charseg_path,model_path_RoI,lineseg_path)
for i in range(n_iteration):
    print(i)
    output_path=para.data_result_path+'/iterate/'+str(i)+'.png'
    evaluate_vote(img_path,output_path,model_path_class,lineseg_path,charseg_path)
    img_path=output_path
'''