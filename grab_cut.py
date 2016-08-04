#!/usr/bin/env python

import numpy as np
import cv2
import sys

BLUE  = [255,0,0]       # rectangle color
RED   = [0,0,255]       # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG    = {'color' : BLACK, 'val' : 0}
DRAW_FG    = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED,   'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
mask_flag = None # TODO Necessary?
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,mask_flag,ix,iy,rect_over

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            mask_flag = False

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        mask_flag = False
        print " Now press the key 'n' a few times until no further change \n"

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print "first draw rectangle \n"
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

def exit_app():
    cv2.destroyAllWindows()
    exit()

if len(sys.argv) == 2:
    filename = sys.argv[1] # for drawing purposes
else:
    exit_app()

img = cv2.imread(filename)
img2 = img.copy()                               # a copy of original image
mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
output = np.zeros(img.shape,np.uint8)           # output image to be shown

# input and output windows
cv2.namedWindow('output')
cv2.namedWindow('input')
cv2.setMouseCallback('input',onmouse)
cv2.moveWindow('input',img.shape[1]+10,90)

while(1):
    cv2.imshow('output',output)
    cv2.imshow('input',img)
    k = 0xFF & cv2.waitKey(1)

    # key bindings
    if k == ord('q'):
        exit_app()
    elif k == ord('0'): # BG drawing
        print " mark background regions with left mouse button \n"
        value = DRAW_BG
    elif k == ord('1'): # FG drawing
        print " mark foreground regions with left mouse button \n"
        value = DRAW_FG
    elif k == ord('2'): # PR_BG drawing
        value = DRAW_PR_BG
    elif k == ord('3'): # PR_FG drawing
        value = DRAW_PR_FG
    #elif k == ord('s'): # save image
    #    bar = np.zeros((img.shape[0],5,3),np.uint8)
    #    res = np.hstack((img2,bar,img,bar,output))
    #    cv2.imwrite('grabcut_output.png',res)
    #    print " Result saved as image \n"
    #elif k == ord('r'): # reset everything
    #    print "resetting \n"
    #    rect = (0,0,1,1)
    #    drawing = False
    #    rectangle = False
    #    mask_flag = True
    #    rect_over = False
    #    value = DRAW_FG
    #    img = img2.copy()
    #    mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
    #    output = np.zeros(img.shape,np.uint8)           # output image to be shown
    elif k == ord('n'): # segment the image
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)

        if mask_flag:
            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
        else:
            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
            mask_flag = True

    mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
    output = cv2.bitwise_and(img2,img2,mask=mask2)
