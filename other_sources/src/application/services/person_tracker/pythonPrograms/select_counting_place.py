#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#from Tkinter import *
#from PIL import Image
#from PIL import ImageTk
import cv2
import numpy as np
import os
import json
import base64

class SelectCountingPlase():

    def __init__(self,imageB64):
        width = 320
        height = 240 
        im_bytes = base64.b64decode(imageB64)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        image = cv2.resize(image, (width, height),
		                       interpolation=cv2.INTER_LINEAR)
        self.points=[]
        self.roi=[]
        self.color=(0,0,255)
        self.im=image
        self.imgr=image
        self.path = '/app/service/src/application/services/person_tracker/cameraConfig/'


    def draw_Roi_callback(self,event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            if len(self.points)<2:
                self.points.append((x,y))

        elif event==cv2.EVENT_LBUTTONUP:
            if len(self.points)==2:
                cv2.line(self.im,(self.points[-2][0],self.points[-2][1]),(self.points[-1][0],self.points[-1][1]),self.color,3)
                data = {}
                data['points']=self.points
                with open(self.path +'config.json', 'w') as outfile:
                    json.dump(data, outfile)

    def select_points(self):

        file = self.path + 'config.json'
        print(self.path)
        if not os.path.exists(file):
            cv2.namedWindow("selection 2 points")
            cv2.setMouseCallback('selection 2 points',self.draw_Roi_callback)
            while(1):
                cv2.imshow('selection 2 points',self.im)
                k = cv2.waitKey(1) & 0xFF
                if (k == ord("q")):  # q is pressed
                    break

            data = {}
            data['points']=self.points
            with open(self.path +'config.json', 'w') as outfile:
                json.dump(data, outfile)
            cv2.release()
            cv2.destroyAllWindows()
        else:
            print ("esa camara ya fue configurada")
            with open(self.path +'config.json') as json_file:
                data = json.load(json_file)
            self.points = data['points']




if __name__ == "__main__":

    im = cv2.imread("image.jpg")
    #rois=Sel_Streets(im)
    #rois.streets_rois()
