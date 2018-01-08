

import cv2
import numpy as np
import tensorflow as tf
import time
import statistics
import h5py
vid_file = '/home/vijayaganesh/Videos/Google Chrome Dinosaur Game [Bird Update] BEST SCORE OF THE WORLD (No hack).mp4'
data_file = 'training_data.txt'
roi_x = 320
roi_y = 120
roi_w = 459
roi_h = 112
font = cv2.FONT_HERSHEY_SIMPLEX
vid = cv2.VideoCapture(vid_file)
### jump Case
jx = 0
jy = 48
jw = 30
jh = 40
# tx = 0
# ty = 30
# tw = 30
# th = 41

### Duck Case
dx = 0
dy = 102
dw = 45
dh = 10

### Idle Case
tx = 0
ty = 68
tw = 30
th = 27

### Variables to store state of jump
prev_j = ty

### Obstacle List

# prev_j_1 = ty
dist = 500
prev_dist = 500
frame_count = 1
speed_list = list()

speed = 0
dino_y = 0
control = ''
file = open(data_file,'w')
while(vid.isOpened()):
    _,frame = vid.read()
    roi_rgb = frame[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
    roi = cv2.cvtColor(roi_rgb,cv2.COLOR_BGR2GRAY)
    print(frame.shape[:2])
    _,roi_thresh = cv2.threshold(roi,150,255,cv2.THRESH_BINARY_INV)
    _,contours,hierarchy = cv2.findContours(roi_thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obstacle_x,obstacle_y = 500,500
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if(w < 7 and h < 7):
            continue
        if(x > 340 and y == 4 ):
            continue
        if(x == jx and w ==jw):
            if(prev_j-y > 0  and y < 67 and y > 45):
                control = 'u'
            prev_j = y
            dino_y = y
        elif(x == dx and y == dy and w == dw and h == dh):
            control = 'd'
            dino_y = y
        elif(x == dx):
            control = 'na'
            dino_y = y
        if(x>40):
            cv2.rectangle(frame,(x+roi_x,y+roi_y),(roi_x+x+w,roi_y+y+h),(0,255,0),2)
            if(x<obstacle_x):
                obstacle_x = x;
                obstacle_y = y;
    dist = obstacle_x
    cv2.putText(frame,'x = '+repr(obstacle_x)+","+repr(obstacle_y),(10,600), font, 4,(255,0,0),2,cv2.LINE_AA)
    if(frame_count < 30):
        speed_list.append(prev_dist - dist)
    else:
        speed = max(speed_list,key=speed_list.count)
        speed_list = list()
        frame_count = 0
    cv2.putText(frame,repr(dino_y),(10,400), font, 4,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,control,(10,500), font, 4,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,'dx/dt = '+repr(speed),(10,700), font, 4,(255,0,0),2,cv2.LINE_AA)
    prev_dist = dist
    file.write(repr(dino_y)+","+repr(speed)+","+repr(obstacle_x)+","+repr(obstacle_y)+","+control+"\n")
    cv2.imshow('roi',frame)
#     time.sleep(0.1)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()
