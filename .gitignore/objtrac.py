"""
Created on Wed Jun  6 11:54:57 2018

@author: swarn4399
"""

import cv2
import numpy as np
from collections import deque #deque is double ended queue
cap=cv2.VideoCapture() #to interact with PC camera
cap.open(0)            #0 for internal camera, 1 for external camera
pts=deque(maxlen=32)
while cap.isOpened(): #camera open or not
    #Reading the frame
    ret,frame=cap.read() #image array captured, ret stores image array, ret is TRUE or return 1 implies shot capture, 0 implies no capture
    #alternative
    #_,frame=cap.read()
    frame=cv2.flip(frame,flipCode=1)# 1 signifies Flip along Y axis
    #Changing the colour space
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #Creating the mask
    lower_red=np.array([160,100,100])
    upper_red=np.array([179,255,255])
    mask=cv2.inRange(hsv,lower_red,upper_red) #inRange is like an if-else statement
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #Finding the Contour
    contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2] #EXTERNAL for only the boundary, APPROX for only the approximate shape
    center=None
    
    if len(contours)>0:
        c = max(contours, key=cv2.contourArea) #contour of maximum area
        ((x,y),rad)=cv2.minEnclosingCircle(c)  #return the minimum circle area that can enclose the contour
        M=cv2.moments(c)                       #determine MI of the circular section
        center=(int(M['m10']/M['m00']),int(M['m01']/M['m00'])) #10 for X-axis area, 01 for Y-axis area, 00 for original area
        if(rad>10):
            cv2.circle(frame, (int(x), int(y)), int(rad),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), cv2.LINE_AA)
    cv2.imshow("MASK",mask)
    cv2.imshow("FRAME",frame)
    k=(cv2.waitKey(30)&0xFF)   #waits for 30ms for user input, if no input it skips
    if(k==ord('q')):           #if k = ASCII value of q, it will reset the q
        pts=deque(maxlen=32)
    if(k==27):                 #27 for escape key
        break
cv2.destroyAllWindows()       
cap.release()             #while in use no other program can use camera, so releases the camera for other programs

#cv2.imwrite('file name',image)
