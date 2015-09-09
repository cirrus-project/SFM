import cv2
import numpy as np
import os,sys
import math as m

cap = cv2.imread(filename)
cap.set(cv2.CAP_PROP_POS_FRAMES,start)
S = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(outputfilename, cv2.VideoWriter_fourcc('M','J','P','G'), cap.get(cv2.CAP_PROP_FPS), S, True)
 

warp_mode = cv2.MOTION_EUCLIDEAN
warp_matrix = np.eye(2, 3, dtype=np.float32)
number_of_iterations = 200;
 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-4;
 
# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 

im1_gray = np.array([])
first = np.array([])
for tt in range(frames):
    # Capture frame-by-frame
    _, frame = cap.read()
    if not(im1_gray.size):
        im1_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        first = frame.copy()
    im2_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    im2_aligned = cv2.warpAffine(frame, warp_matrix, (S[0],S[1]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
 
    gsout =  cv2.cvtColor(im2_aligned,cv2.COLOR_BGR2GRAY)
    im2_aligned[gsout<100,:]=first[gsout<100,:]
    out.write(im2_aligned)
cap.release()
out.release()



