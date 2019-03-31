from PIL import Image
from random import random,randint
import numpy as np
import scipy.misc
import math

def gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussDrop(im0,drop_rate=0.1,filter_size=(7,7),sigma=1):
    output=np.zeros((40,19),dtype=np.uint8)
    filter=gauss2D(filter_size,sigma)
    verti_margin=math.floor(filter_size[0]/2)
    hori_margin=math.floor(filter_size[1]/2)
    filter=filter/filter[verti_margin][hori_margin]*1.5
    #print(filter)
    for i in range(40):
        for j in range(19):
            output[i][j]=im0[i][j]
    for i in range(40):
        for j in range(19):
            if random()<drop_rate:
                for u in range(filter.shape[0]):
                    for v in range(filter.shape[1]):
                        hori=u-verti_margin
                        verti=v-hori_margin
                        if i+u>=0 and i+u<40 and j+v>=0 and j+v<19:
                            if random()<filter[u][v]:
                                output[i+u][j+v]=min(255,im0[i+u][j+v]+255*filter[u][v])
    return output

def Dilation(im0):
    output=np.zeros((40,19),dtype=np.uint8)
    for i in range(40):
        for j in range(19):
            output[i][j]=im0[i][j]
    for i in range(40):
        for j in range(19):
            if im0[i][j]<255:
                if i-1>0:
                    output[i-1][j]=min(output[i-1][j],im0[i][j])
                if i+1<40:
                    output[i+1][j]=min(output[i+1][j],im0[i][j])
                if j-1>0:
                    output[i][j-1]=min(output[i][j-1],im0[i][j])
                if j+1<19:
                    output[i][j+1]=min(output[i][j+1],im0[i][j])
    return output

def Shift(im0,x,y):
    output=np.zeros((40,19),dtype=np.uint8)+255
    for i in range(40):
        for j in range(19):
            if i+x>=0 and i+x<40 and j+y>=0 and j+y<19:
                output[i+x][j+y]=im0[i][j]
    return output

def Degrade(im0):#may not degrade
    output=im0
    for i in range(randint(0,2)):
        output=Dilation(output)
    if randint(0,5)>0:
        output=gaussDrop(output)
    output=Shift(output,randint(-3,3),randint(-3,3))
    return output
'''
im0 = Image.open(para.data_result_path+'/data\healthy\patches_healthy\e.png')
im0 = np.array(im0)
for i in range(100):
    output=Degrade(im0)
    img = Image.fromarray(output, 'L')
    img.save(para.data_result_path+'/simulate/simulate'+str(i)+'.png')
'''
