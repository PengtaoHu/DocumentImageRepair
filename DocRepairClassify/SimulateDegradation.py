from PIL import Image
from random import random,randint
import numpy as np
import scipy.misc
import math
import para

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
    output=np.zeros(para.patch_size,dtype=np.uint8)
    filter=gauss2D(filter_size,sigma)
    verti_margin=math.floor(filter_size[0]/2)
    hori_margin=math.floor(filter_size[1]/2)
    filter=filter/filter[verti_margin][hori_margin]*1.5
    for i in range(40):
        for j in range(19):
            output[i][j]=im0[i][j]
    for i in range(40):
        for j in range(19):
            if random()<drop_rate:
                for u in range(filter.shape[0]):
                    for v in range(filter.shape[1]):
                        if i+u>=0 and i+u<40 and j+v>=0 and j+v<19:
                            if random()<filter[u][v]:
                                output[i+u][j+v]=max(output[i+u][j+v],min(255,im0[i+u][j+v]+255*filter[u][v]))
    return output

def Dilation(im0,rate=1):
    output=np.zeros(para.patch_size,dtype=np.uint8)
    for i in range(40):
        for j in range(19):
            output[i][j]=im0[i][j]
    for i in range(40):
        for j in range(19):
            if random()<rate and im0[i][j]<255:
                if i-1>0:
                    output[i-1][j]=min(output[i-1][j],im0[i][j])
                if i+1<40:
                    output[i+1][j]=min(output[i+1][j],im0[i][j])
                if j-1>0:
                    output[i][j-1]=min(output[i][j-1],im0[i][j])
                if j+1<19:
                    output[i][j+1]=min(output[i][j+1],im0[i][j])
    return output

def Shift(im0,y_shift,x_shift):
    output=np.zeros(para.patch_size,dtype=np.uint8)+255
    for i in range(40):
        for j in range(19):
            if i+y_shift>=0 and i+y_shift<40 and j+x_shift>=0 and j+x_shift<19:
                output[i+y_shift][j+x_shift]=im0[i][j]
    return output

def Degrade(im0,y_shift=None,x_shift=None):#may not degrade
    output=im0

    dilation_rate=0
    for i in range(randint(0,2)):
        if randint(0,5)==0:
            output=Dilation(output)
            dilation_rate+=1
        else:
            k=0.3+0.7*random()
            output=Dilation(output,k)
            dilation_rate+=k

    if(randint(0,5)>0):
        output=gaussDrop(output,drop_rate=0.15*random())

    if y_shift is None:
        y_shift=randint(-7,7)
    if x_shift is None:
        x_shift=randint(-4,4)
    output=Shift(output,y_shift,x_shift)

    return output,y_shift,x_shift

def HalfHalfPatch(im_left,im_main,im_right,shift_from_center=0):
    output=np.zeros((para.patch_size[0],para.patch_size[1]*3),dtype=np.uint8)+255
    output[:,para.patch_size[1]:para.patch_size[1]*2]=im_main
    offset_left=randint(0,5)
    offset_right=randint(0,5)
    output[:,para.patch_size[1]*2-offset_right:para.patch_size[1]*3-offset_right]=np.minimum(output[:,para.patch_size[1]*2-offset_right:para.patch_size[1]*3-offset_right],im_right)
    output[:,offset_left:para.patch_size[1]+offset_left]=np.minimum(output[:,offset_left:para.patch_size[1]+offset_left],im_left)
    output=output[:,para.patch_size[1]+shift_from_center:para.patch_size[1]*2+shift_from_center]
    output=Degrade(output,x_shift=0)
    return output

if __name__ == '__main__':
    im0 = Image.open(para.data_result_path+'/data\healthy\patches_healthy/s.png')
    im0 = np.array(im0)
    npz_healthy = np.load(para.data_result_path+'/healthy.npz')
    healthy_labels=npz_healthy['labels']
    healthy_patches=npz_healthy['patches']
    for i in range(100):
        output=SegBarPatch(healthy_patches[randint(0,95)],healthy_patches[randint(0,95)])[0]
        img = Image.fromarray(output, 'L')
        img.save(para.data_result_path+'/simulate/simulate'+str(i)+'.png')

