from SimulateDegradation import Degrade,Shift
from random import randint
import para
import Image
import numpy as np
import cv2 as cv

def DiffScore(im0, im1,d_max=3):#TODO: deal with space
    score=0
    count=0
    #np.pad(im0,(d_max,d_max),'constant',constant_values=(0, 0))
    #np.pad(im1,(d_max,d_max),'constant',constant_values=(0, 0))
    for i in range(40):
        for j in range(19):
            if im0[i][j]!=0:
                continue
            flag=0
            count+=1
            for d in range(d_max):
                for u in range(d+1):
                    v=d-u
                    if( i+u<40 and j+v<19 and im0[i][j]==im1[i+u][j+v] or 
                        i+u<40 and j-v>=0 and im0[i][j]==im1[i+u][j-v] or 
                        i-u>=0 and j+v<19 and im0[i][j]==im1[i-u][j+v] or 
                        i-u>=0 and j-v>=0 and im0[i][j]==im1[i-u][j-v]):
                        score+=d*d
                        flag=1
                        break
                if flag==1 :
                    break
            if flag==0:
                score+=d_max*d_max
    if count==0:
        return d_max
    return score/count

def SearchAlign(im0,im1,x_limit=4,y_limit=4):
    th0, im0 = cv.threshold(im0,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    th1, im1 = cv.threshold(im1,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    score_min=float('inf')
    for i in range(-x_limit,x_limit+1):
        for j in range(-y_limit,y_limit+1):
            score=(DiffScore(im0, Shift(im1, j,i))
                   +DiffScore(Shift(im1, j,i),im0))/2
            if score<score_min:
                score_min=score
                x_shift=i
                y_shift=j

    '''
    for i in range(-x_limit,x_limit+1):
        score=(DiffScore(im0, Shift(im1, 0,i))+DiffScore(Shift(im1, 0,i),im0))/2
        if score<score_min:
            score_min=score
            x_shift=i
    y_shift=0
    for i in range(-y_limit,y_limit+1):
        score=(DiffScore(im0, Shift(im1, i,x_shift))+DiffScore(Shift(im1, i,x_shift),im0))
        if score<score_min:
            score_min=score
            y_shift=i
    '''
    return (score_min,y_shift,x_shift)

def main():
    npz_healthy = np.load(para.data_result_path+'/healthy.npz')
    healthy_labels=npz_healthy['labels']
    healthy_patches=npz_healthy['patches']
    for i in range(1000):
        f = open(para.data_result_path+'/verify_results/'+str(i)+'.txt', 'w')
        idx=randint(0,len(healthy_patches)-1)
        (im0,y_shift,x_shift)=Degrade(healthy_patches[idx])
        f.write(str(healthy_labels[idx])+' '+str(y_shift)+' '+str(x_shift)+'\r\n\r\n')
        img= Image.fromarray(im0,'L')
        img.save(para.data_result_path+'/verify_results/'+str(i)+'.png')
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
        img.save(para.data_result_path+'/verify_results/'+str(i)+'_match.png')
        for i in min_scores:
            f.write(str(scores[i])+' '+str(healthy_labels[i])+' '+str(y_shifts[i])+' '+str(x_shifts[i])+'\r\n')
        f.write('\r\n')
        for i in range(len(scores)):
            f.write(str(scores[i])+' '+str(healthy_labels[i])+' '+str(y_shifts[i])+' '+str(x_shifts[i])+'\r\n')
        f.close()

if __name__ == '__main__':
    main()
