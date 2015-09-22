
import cv2
import numpy as np
import os,sys
import math as m
import pandas as pd

#
im1 = cv2.imread('IMG_3954.JPG',0)
im2 = cv2.imread('IMG_3955.JPG',0)
#
number_of_iterations = 2100;
# 
#            # Specify the threshold of the increment
#            # in the correlation coefficient between two iterations
termination_eps = 1e-6;
# 
#            # Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
warp_matrix = np.eye(3, 3, dtype=np.float32)
warp_mode = cv2.MOTION_HOMOGRAPHY
(cc,rigid_mat) = cv2.findTransformECC(im1,im2,warp_matrix,warp_mode,criteria)


im3 = cv2.warpPerspective(im2, rigid_mat,(im2.shape[1],im2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


im31 = im3[1700:2000,1200:1800].copy()
cv2.imwrite('test3.png',im31)

im11 = im1[1700:2000,1200:1800].copy()
cv2.imwrite('test4.png',im11)
