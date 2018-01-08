import pyscreenshot as ImageGrab
import numpy as np
import cv2
import time
from keras.models import load_model
import os
from pynput.keyboard import Key, Controller
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

keyboard = Controller()

roi_x = 320
roi_y = 120
roi_w = 459
roi_h = 112

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

model = load_model('keras_model.h5')

def rangeConverter(x,x_min,x_max,y_min,y_max):
    y = (((x - x_min) * (y_max-y_min))/ (x_max-x_min)) + y_min
    return y
#     img = ImageGrab.grab(bbox=(0,0,1920,1080)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
#     img_np = np.array(img) #this is the array obtained from conversion
while(True):
    t = time.time()
    os.system("scrot -q 10 temp.jpeg")
    frame = cv2.imread("temp.jpeg")
    img = cv2.resize(frame,(1114,720),cv2.INTER_AREA)
    roi_rgb = img[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w]
    roi = cv2.cvtColor(roi_rgb,cv2.COLOR_BGR2GRAY)
    cv2.imshow('image',roi)
#     print(frame.shape[:2])
    _,roi_thresh = cv2.threshold(roi,150,255,cv2.THRESH_BINARY_INV)
    _,contours,hierarchy = cv2.findContours(roi_thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obstacle_x,obstacle_y = 500,500
    print(len(contours))
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if(w < 7 and h < 7):
            continue
        if(x > 340 and y == 4 ):
            continue
        if(x == jx and w ==jw):
            if(prev_j-y > 0  and y < 67 and y > 45):
            prev_j = y
            dino_y = y
        elif(x == dx and y == dy and w == dw and h == dh):
            dino_y = y
        elif(x == dx):
            dino_y = y
        if(x>40):
            if(x<obstacle_x):
                obstacle_x = x;
                obstacle_y = y;
    dist = obstacle_x
    if(frame_count < 30):
        speed_list.append(prev_dist - dist)
    else:
        speed = max(speed_list,key=speed_list.count)
        speed_list = list()
        frame_count = 0
    prev_dist = dist
    output = model.predict(np.array([[dino_y,speed,obstacle_x,obstacle_y]]))
    if(output == array([2])):
        keyboard.press('up')
        keyboard.release('up')
    elif(output == array([3])):
        keyboard.press('down')
        keyboard.release('down')
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
