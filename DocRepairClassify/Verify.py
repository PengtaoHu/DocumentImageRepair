from SimulateDegradation import Degrade,Shift
from random import randint
import para
import Image
import numpy as np
import cv2 as cv
from scipy import ndimage

def DiffScore(im0, im1,d_max=5):
    im0=1-im0
    count=np.sum(im0)
    im1=ndimage.distance_transform_edt(im1)
    im1=np.minimum(im1,d_max)
    im1=np.multiply(im1,im1)
    score=np.sum(np.multiply(im0,im1))
    if count==0:
        return 0
    return score/count

def SearchAlign(im0,im1,x_limit=4,y_limit=4):
    th0, im0 = cv.threshold(im0,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
    th1, im1 = cv.threshold(im1,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
    score_min=float('inf')
    for i in range(-x_limit,x_limit+1):
        for j in range(-y_limit,y_limit+1):
            score=(DiffScore(im0, Shift(im1, j,i,1))
                   +DiffScore(Shift(im1, j,i,1),im0))/2
            if score<score_min:
                score_min=score
                x_shift=i
                y_shift=j

    return (score_min,y_shift,x_shift)

def main():
    npz_healthy = np.load(para.data_result_path+'/healthy.npz')
    healthy_labels=npz_healthy['labels']
    healthy_patches=npz_healthy['patches']
    for i in range(100):
        f = open(para.data_result_path+'/verify_test/'+str(i)+'.txt', 'w')
        idx=randint(0,len(healthy_patches)-1)
        (im0,y_shift,x_shift)=Degrade(healthy_patches[idx],y_shift=randint(-4,4),x_shift=randint(-4,4))
        f.write(str(healthy_labels[idx])+' '+str(y_shift)+' '+str(x_shift)+'\r\n\r\n')
        img= Image.fromarray(im0,'L')
        img.save(para.data_result_path+'/verify_test/'+str(i)+'.png')
        label0=healthy_labels[idx]
        scores=[]
        x_shifts=[]
        y_shifts=[]
        for im1 in healthy_patches:
            result=SearchAlign(im0,im1)
            scores.append(result[0])
            y_shifts.append(result[1])
            x_shifts.append(result[2])
        min_scores=np.argsort(scores)[:3]
        img= Image.fromarray(Shift(healthy_patches[min_scores[0]],y_shifts[min_scores[0]],x_shifts[min_scores[0]]),'L')
        img.save(para.data_result_path+'/verify_test/'+str(i)+'_match.png')
        for i in min_scores:
            f.write(str(scores[i])+' '+str(healthy_labels[i])+' '+str(y_shifts[i])+' '+str(x_shifts[i])+'\r\n')
        f.write('\r\n')
        for i in range(len(scores)):
            f.write(str(scores[i])+' '+str(healthy_labels[i])+' '+str(y_shifts[i])+' '+str(x_shifts[i])+'\r\n')
        f.close()

if __name__ == '__main__':
    main()
