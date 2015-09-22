
import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

#
#im1 = cv2.imread('IMG_3954.JPG',0)
#im2 = cv2.imread('IMG_3955.JPG',0)
#
#number_of_iterations = 2100;
# 
#            # Specify the threshold of the increment
#            # in the correlation coefficient between two iterations
#termination_eps = 1e-6;
# 
#            # Define termination criteria
#criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
#warp_matrix = np.eye(2, 3, dtype=np.float32)
#warp_mode = cv2.MOTION_TRANSLATION
#(cc,rigid_mat) = cv2.findTransformECC(im1,im2,warp_matrix,cv2.MOTION_TRANSLATION,criteria)
#im3 = cv2.warpAffine(im2, rigid_mat,(im2.shape[1],im2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#cv2.imwrite('test2.png',im3)
#
#comb = cv2.addWeighted(im3,0.5,im1,0.5,0.0)
#comb = cv2.absdiff(im3,im1)
#
number_of_iterations = 2000;
# 
#            # Specify the threshold of the increment
#            # in the correlation coefficient between two iterations
termination_eps = 1e-1;
# 
#            # Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
#
#im31 = im3[1700:1950,1200:1450].copy()
#im11 = im1[1700:1950,1200:1450].copy()
im31 = cv2.imread('test3.png',0)
im11 = cv2.imread('test4.png',0)


box = 8
nx = im11.shape[0]
ny = im11.shape[1]

kernel = np.ones((2*box,2*box),np.float32)/(2*box*2*box)
avIm11 = cv2.filter2D(im11,-1,kernel)
avIm31 = cv2.filter2D(im31,-1,kernel)

im4 = 125*np.ones_like(im11).astype(int)
im45 = 125*np.ones_like(im11).astype(float)
for i in range(2*box,nx-2*box):
    print(i)

    for j in range(2*box,ny-2*box):
#        subIm1 = cv2.resize(im11[i-box:i+box,j-box:j+box].copy(),(0,0),fx=8,fy=8,interpolation=cv2.INTER_LINEAR)
#        subIm2 = cv2.resize(im31[i-box:i+box,j-box:j+box].copy(),(0,0),fx=8,fy=8,interpolation=cv2.INTER_LINEAR)
#        warp_matrix = np.eye(2, 3, dtype=np.float32)
#        try:
#            (cc,warp_matrix) = cv2.findTransformECC(subIm1,subIm2,warp_matrix,cv2.MOTION_TRANSLATION, criteria)
#        except:
#            warp_matrix = np.eye(2, 3, dtype=np.float32)
#        im4[i,j]=int(255.0*((warp_matrix[0,2]**2 + warp_matrix[1,2]**2)**0.5)/15.0)
#        continue
#i=20
#j=20
        subIm1 = im11[i-box:i+box,j-box:j+box].copy()

#cv2.imshow('window',subIm1)
#cv2.waitKey(0)

        meanIm1 = avIm11[i,j]
        maxCV = 0
        distance = 0
        maxJ = 0
        for offI in range(i-box,i+box):
            for offJ in range(j-box,j+box):
                subIm2 = im31[offI-box:offI+box,offJ-box:offJ+box].copy()
                thisCV = (np.average(np.multiply(subIm1.astype(int),subIm2.astype(int))) - np.average(subIm1)*np.average(subIm2))/(np.std(subIm1.astype(int))*np.std(subIm2.astype(int)))
        
                if thisCV>maxCV:
                    maxCV = thisCV
                    distance = ((i-offI)**2+(j-offJ)**2)**0.5
                    s2 = subIm2.copy()
#        print(offI,offJ,thisCV)
        #cv2.imshow('window',np.hstack((subIm1,s2)))
        #cv2.waitKey(1)

            
        
       # resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)



        #warp_matrix = np.eye(2, 3, dtype=np.float32)
        #(cc,warp_matrix) = cv2.findTransformECC(subIm1,subIm2,warp_matrix,cv2.MOTION_TRANSLATION, criteria)
        im4[i,j]=int(255.0*distance/box) #int(255.0*((warp_matrix[0,2]**2 + warp_matrix[1,2]**2)**0.5)/5.0)
        im45[i,j]=distance #int(255.0*((warp_matrix[0,2]**2 + warp_matrix[1,2]**2)**0.5)/5.0)
        #print(distance,im4[i,j])
        #im4[i,j]=int(255.0*cc)
        #print((int(255.0*((warp_matrix[0,2]**2 + warp_matrix[1,2]**2)**0.5)/5.0)))


        

cv2.imwrite('test.png',im4)
#cv2.namedWindow('window', flags =  cv2.WINDOW_AUTOSIZE)
#cv2.imshow('window',subIm1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
