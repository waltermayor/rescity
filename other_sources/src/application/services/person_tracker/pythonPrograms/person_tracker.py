#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""@author: walter
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip
from collections import deque
#from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
import sys, os
import cv2
import math
import simplejson
from random import randint

from . import helpers
from . import detector
from . import tracker
from . import select_counting_place


#import helpers
#import detector
#import tracker
#import select_counting_place

import copy
import time
import pylab
#import structures



class PersonTracker():

    frame_count = 0 # frame counter
    max_age = 5 #12  # no.of consecutive unmatched detection before
        # a track is deleted
    min_hits =2  # no. of consecutive matches needed to establish a track
    scale=np.float(146/99)
    frameRate =24#30
    pathVideo=""
    streetList=[]
    global local
    tracker_list =[] # list for trackers
    points = []

    # list for track ID
    track_id_list= deque(['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK','AL','AM',
            'AN','AO','AP','AQ','AR','AS','AT','AU','AV','AW','AX','AY','AZ',
            'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK','BL','BM',
            'BN','BO','BP','BQ','BR','BS','BT','BU','BV','BW','BX','BY','BZ',
            'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK','CL','CM',
            'CN','CO','CP','CQ','CR','CS','CT','CU','CV','CW','CX','CY','CZ',
            'DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DJ', 'DK','DL','DM',
            'DN','DO','DP','DQ','DR','DS','DT','DU','DV','DW','DX','DY','DZ',
            'EA', 'EB', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EJ', 'EK','EL','EM',
            'EN','EO','EP','EQ','ER','ES','ET','EU','EV','EW','EX','EY','EZ',
            'FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI', 'FJ', 'FK','FL','FM',
            'FN','FO','FP','FQ','FR','FS','FT','FU','FV','FW','FX','FY','FZ'])

    debug = False #True
    withImages=True
    withDarknet = True
    velConstant = True # cambiar tambien en el tracker.py
    clips = []
    number_of_close_persons: int = 0 
    number_of_persons: int = 0
    persons_without_mask: int = 0

    def __init__(self):
        self.det = detector.CarDetector(self.withDarknet)


    def beginAnalisisVideo(self,arg):

        start=time.time()   
        nvid = os.path.splitext(arg)[0]	

        if nvid=="camaraTunel1":
            local=1
        else:
            local=2

        voutput = '/home/walter/WorkSpace/Rescity/otrasFuentes/Program/videoOutputs/o' + nvid + '.mp4'
        vinput = '/home/walter/WorkSpace/Rescity/otrasFuentes/Program/videoInputs/' + arg
        #fps = 30
        #cap = cv2.VideoCapture(random)
        #cap = cv2.VideoCapture(vinput)
        #output = 'pruebas/salida11.mp4'
        pathVideo= nvid #"camara4"
        clip1 = VideoFileClip(vinput)#.subclip(4,49) # The first 8 seconds doesn't have any cars...
        clip1=clip1.set_fps(self.frameRate)
        clip = clip1.fl_image(self.pipeline)
        clip.write_videofile(voutput, audio=False,fps=self.frameRate)
        end  = time.time()
        print(round(end-start, 2), 'Seconds to finish')

    def beginAnalisisImage(self,file):

        
        img = self.pipeline(file)

        return img

    def analyseImage(self,file):

        img = self.pipeline(file)

        return self.number_of_persons, self.persons_without_mask, self.number_of_close_persons

    def box_iou2(self, a, b):
        '''
        Helper funciton to calculate the ratio between intersection and the union of
        two boxes a and b
        a[0], a[1], a[2], a[3] <-> left, up, right, bottom
        '''
        w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
        h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
        s_intsec = w_intsec * h_intsec
        s_a = (a[2] - a[0])*(a[3] - a[1])
        s_b = (b[2] - b[0])*(b[3] - b[1])

        if((s_a + s_b -s_intsec)>0):
            return float(s_intsec)/(s_a + s_b -s_intsec)
        else:
            return float(s_intsec)/0.00000000000000000000000000000001

        
    def assign_detections_to_trackers(self, trackers, detections, iou_thrd = 0.3):

        IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
        for t,trk in enumerate(trackers):
            for d,det in enumerate(detections):
                IOU_mat[t,d] = self.box_iou2(trk,det)

        # Hungarian algorithm or Munkres algorithm

        matched_idx = linear_assignment(-IOU_mat)

        unmatched_trackers, unmatched_detections = [], []
        #row_idx, col_idx = matched_idx
        for t,trk in enumerate(trackers):
            if(t not in matched_idx[0]):
                unmatched_trackers.append(t)

        
        for d, det in enumerate(detections):
            if(d not in matched_idx[1]):
                unmatched_detections.append(d)

        matches = []

        # For creating trackers we consider any detection with an
        # overlap less than iou_thrd to signifiy the existence of
        # an untracked object

        match = matched_idx
        for m in range(len(matched_idx[0])):
            if(IOU_mat[match[0][m],match[1][m]]<iou_thrd):
                unmatched_trackers.append(match[0][m])
                unmatched_detections.append(match[1][m])
            else:
                #matches.append(m.reshape(1,2))  array([[2, 8]]))
                matches.append(np.array([[match[0][m], match[1][m]]]))

        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def calculate_centroid(self, xx,tmp_trk):
        cx=xx[1]+((xx[3]-xx[1])/2)
        cy=xx[0]+((xx[2]-xx[0])/2)
        centroid=np.array([cx,cy])

        if len(tmp_trk.centroids)>1:
            slwindow=(tmp_trk.centroids[-1]+centroid)/2
            centroid=slwindow

        centroidIni=np.expand_dims(np.array([centroid[0],centroid[1],1]), axis=0).T
        centroid_old=tmp_trk.centroids[len(tmp_trk.centroids)-1]
        distancePix=np.float(np.sqrt(sum((centroid-centroid_old)**2)))

        success=0

        if tmp_trk.matched[-1]==1:
            tmp_trk.centroids.append(centroid)
            tmp_trk.frameNum.append(self.frame_count)
            success=1

        return tmp_trk,success,distancePix



    def calculate_number_of_close_persons(self, trk):
        self.number_of_close_persons = 0
        if len(self.tracker_list)!=0:

            centTrkOne = trk.centroids[-1]
            print("entramos")
            for idx,otherTrk in enumerate(self.tracker_list):
                centTrkTwo = otherTrk.centroids[-1]
                dist=np.float(np.sqrt(sum((centTrkOne-centTrkTwo)**2)))
                #print("Distance",dist)
                
                if dist < 5:
                    #trk.colorProb=[255,0,0]
                    self.number_of_close_persons +=1
                else:
                    pass
                    #trk.colorProb=[0,0,255]
        else:
            trk.crossPoint.append(200)
        return trk

    def calculate_number_of_persons(self, trk):
       
        centTrkOne = trk.centroids[-1]
        point1 = self.points[0]
        point2 = self.points[1]
        dif_x = abs(point2[0]-point1[0])
        dif_y = abs(point2[1]-point1[1])

        if dif_x > dif_y:
            reference_line = point2[1]
            print("hello antes")
            diference = reference_line - centTrkOne[1]
            if diference > 0:
                print("entre antes")
                print("dif",diference)
                trk.before_line = True
            elif (diference < 0 and diference > -20) and trk.before_line == False : 
                trk.before_line = True
            elif diference < 0:
                trk.after_line = True
                print("counted",trk.counted)
                print("counted",trk.id)
                print("dif",diference)
                print("entre despues")
        else:
            reference_line = point2[0]
            print("hello x")


        if trk.before_line and trk.after_line and not trk.counted:
            self.number_of_persons +=1
            trk.counted = True
        return trk

    def calculate_persons_without_mask(self,trk,with_mask_box, without_mask_box):
        
        if trk.with_mask == None:
            trb = trk.box
            mask_lis=[]
            without_mask_lis =[]
            list_idx = []
            for mask in with_mask_box:
                wmb = mask
                w_intsec = np.maximum (0, (np.minimum(trb[2], wmb[2]) - np.maximum(trb[0], wmb[0])))
                h_intsec = np.maximum (0, (np.minimum(trb[3], wmb[3]) - np.maximum(trb[1], wmb[1])))
                intsec_area = w_intsec * h_intsec
                mask_area = (wmb[2] - wmb[0])*(wmb[3] - wmb[1])
                ratio = intsec_area/mask_area
                mask_lis.append(ratio)

            if(len(mask_lis)>0):
                list_idx.append(max(mask_lis))
            else:
                list_idx.append(-1)  #not exist

            for wmask in without_mask_box:
                wmb = wmask
                w_intsec = np.maximum (0, (np.minimum(trb[2], wmb[2]) - np.maximum(trb[0], wmb[0])))
                h_intsec = np.maximum (0, (np.minimum(trb[3], wmb[3]) - np.maximum(trb[1], wmb[1])))
                intsec_area = w_intsec * h_intsec
                without_mask_area = (wmb[2] - wmb[0])*(wmb[3] - wmb[1])
                ratio = intsec_area/without_mask_area
                without_mask_lis.append(ratio)

            if(len(without_mask_lis)>0):
                list_idx.append(max(without_mask_lis))
            else:
                list_idx.append(-1)  #not exist
            
            max_value = max(list_idx)
            if list_idx.index(max_value) == 0:
                trk.with_mask = True
                trk.colorProb = (0,0,255)
            elif list_idx.index(max_value) == 1:
                trk.with_mask = False
                trk.colorProb = (255,0,0)
                self.persons_without_mask +=1
        else:
            print("already matched")

        print("persons_without_mask ", self.persons_without_mask)
        return trk


    def pipeline(self, img):
    
        self.frame_count+=1

        if self.frame_count==1:
            print("frame 1")
            sel_counting = select_counting_place.SelectCountingPlase(img)
            sel_counting.select_points()
            self.points = sel_counting.points
        
        if self.withDarknet:
            if self.withImages:
                file = img
                img, z_box, with_mask_box, without_mask_box = self.det.get_localizationYolo(file) # measurement
                image = img
            else:
                image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                cv2.imwrite("image.jpg",image)
                img, z_box, with_mask_box, without_mask_box = self.det.get_localizationYolo('image.jpg') # measurement
        else:
            z_box = self.det.get_localization(img) # measurement


        #img_dim = (img.shape[1], img.shape[0])
        
        if self.debug:
            print('Frame:', self.frame_count)

        x_box =[]
        if self.debug:
            for i in range(len(z_box)):
                img1= helpers.draw_box_label(img, z_box[i],(20,43,20),1,(29,34,56),self.number_of_persons, self.persons_without_mask)
                plt.imshow(img1)
            plt.show()

        if len(self.tracker_list) > 0:
            for trk in self.tracker_list:
                x_box.append(trk.box)

        if self.debug:
            for i in range(len(x_box)):
                img1= helpers.draw_box_label(img, x_box[i],(0,0,255),1,(255,0,0),self.number_of_persons, self.persons_without_mask)
                plt.imshow(img1)
            plt.show()


        matched, unmatched_dets, unmatched_trks \
        = self.assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.1)#0.3
        if self.debug:
            print('Detection:', z_box)
            print('x_box: ', x_box)
            print('matched:', matched)
            print('unmatched_det:', unmatched_dets)
            print('unmatched_trks:', unmatched_trks)


        # Deal with matched detections
        if matched.size >0:
            for trk_idx, det_idx in matched:
                z = z_box[det_idx].copy()
                z = np.expand_dims(z, axis=0).T
                tmp_trk= self.tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                if self.velConstant:
                    xx =[xx[0], xx[2], xx[4], xx[6]]
                else:
                    xx =[xx[0], xx[3], xx[6], xx[9]]

                x_box[trk_idx] = xx
                tmp_trk.box =xx
                tmp_trk.matched
                tmp_trk.matched.append(1)
                tmp_trk.hits += 1


        # Deal with unmatched detections
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                z = z_box[idx].copy()
                z = np.expand_dims(z, axis=0).T
                tmp_trk = tracker.Tracker() # Create a new tracker
                if self.velConstant:
                    x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                else:
                    x = np.array([[z[0], 0, 0, z[1], 0, 0, z[2], 0, 0, z[3], 0, 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                if self.velConstant:
                    xx =[xx[0], xx[2], xx[4], xx[6]]
                else:
                    xx =[xx[0], xx[3], xx[6], xx[9]]

                #-------fist time-------------------
                cx=xx[1]+((xx[3]-xx[1])/2)
                cy=xx[0]+((xx[2]-xx[0])/2)
                centroid=np.array([cx,cy])

                tmp_trk.centroids.append(centroid)
                tmp_trk.frameNum.append(self.frame_count)
                #------------------------------------------------
                tmp_trk.before_line = False
                tmp_trk.after_line = False
                tmp_trk.counted = False
                tmp_trk.with_mask = None
                tmp_trk.box = xx
                tmp_trk.id = self.track_id_list.popleft() # assign an ID for the tracker
                self.tracker_list.append(tmp_trk)
                x_box.append(xx)

        # Deal with unmatched tracks
        if len(unmatched_trks)>0:
            for trk_idx in unmatched_trks:
                tmp_trk = self.tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.matched.append(0)
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                if self.velConstant:
                    xx =[xx[0], xx[2], xx[4], xx[6]]
                else:
                    xx =[xx[0], xx[3], xx[6], xx[9]]

                tmp_trk.box =xx
                x_box[trk_idx] = xx


        # The list of tracks to be annotated
        good_tracker_list =[] 
        blackImage = np.zeros(image.shape)
        height, width = image.shape[:2]
        maxHeight=100
        for trk_idx,trk in enumerate(self.tracker_list):
        #for trk in tracker_list:
            if ((trk.hits >= self.min_hits) and (trk.no_losses <=self.max_age)):
                good_tracker_list.append(trk)

                #--------------collision features-----------------
                xx=trk.box
                trk,success,distancePix=self.calculate_centroid(xx,trk)
                trk = self.calculate_number_of_close_persons(trk)
                trk = self.calculate_number_of_persons(trk)
                trk = self.calculate_persons_without_mask(trk,with_mask_box, without_mask_box)
                print("near",self.number_of_close_persons)
                print("number",self.number_of_persons)
                x_cv2 =trk.box

                if self.debug:
                    print('updated box:', x_cv2)
                img= helpers.draw_box_label(img, x_cv2,trk.box_color,trk.id,trk.colorProb, self.number_of_persons, self.persons_without_mask) # Draw the bounding boxes on the
                #img= helpers.draw_line_traker(img,trk.centroids,trk.box_color)
                img= helpers.draw_poliLine_traker(img,trk.centroids,trk.box_color)
                #blackImage= helpers.draw_centroids(blackImage,trk)
        #-------------------------------------------------------------------

        deleted_tracks = filter(lambda x: x.no_losses >self.max_age, self.tracker_list)

        for trk in deleted_tracks:
                self.track_id_list.append(trk.id)

        tracker_list = [x for x in self.tracker_list if x.no_losses<=self.max_age]

        if self.debug:
            print('Ending tracker_list: ',len(tracker_list))
            print('Ending good tracker_list: ',len(good_tracker_list))

        return img
        #return warped


if __name__ == "__main__":
    obj = PersonTracker()
    obj.beginAnalisisVideo(sys.argv[1])
