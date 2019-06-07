'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import para
img=mpimg.imread(para.data_result_path+'/RoI_results/RoI.png')
plt.figure()
plt.imshow(img) 
plt.show()
'''
import os
path='D:\DataResult\data\healthy\patches_healthy'
for root, dirs, names in os.walk(path):
    for name in names:
        print('\''+name.split('.')[0]+'\':\''+name.split('.')[0]+'\',')