#!python3

import base64
import numpy as np
import cv2
from PIL import Image
import sys, os
from matplotlib import pyplot as plt
import time
from glob import glob

#pathw= '/detection/darknet/Program/'
pathw = '/app/service/src/application/services/person_tracker/'
sys.path.append(pathw + 'CNN')
import darknet as dn
import numpy as np



class CarDetector(object):

    def __init__(self, withDarknet):
        self.person_boxes = []
        self.withMask_boxes = []
        self.withoutMask_boxes = []
        self.withKalman=True

        if withDarknet:

            self.network, self.class_names, self.class_colors = dn.load_network( 
                            pathw + "CNN/yolov3.cfg",
      				        pathw + "CNN/coco.data",
       				        pathw + "CNN/yolov3.weights", batch_size= 1)
            self.boxes =[]
            self.scores=[]
            self.classes=[]
            self.num_detections=[]

    # Helper function to convert image into numpy array

    def image_detection(self,image_path, network, class_names, class_colors, thresh):
        # Darknet doesn't accept numpy images.
        # # Create one with image we reuse for each detect
        width = 320 #dn.network_width(network) 
        height = 240 # dn.network_height(network)
        darknet_image = dn.make_image(width, height, 3)
        
        im_bytes = base64.b64decode(image_path)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        image_rgb = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
	    #image = cv2.imread(image_path)
	    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
		                       interpolation=cv2.INTER_LINEAR)
        
        dn.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = dn.detect_image(network, class_names, darknet_image, thresh=thresh)
        dn.free_image(darknet_image)
        image = dn.draw_boxes(detections, image_resized, class_colors)
        cv2.imwrite("image.jpg",image)
        #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections
        return image_resized, detections


    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
        #para trabajar con el detector de tensorflow y el traker filtro de kalman
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)

    def box_normal_to_pixel2(self, box):
        # para trabajar con el detector yolo y traker filtro de kalman
        box_pixel=[int(box[1]-(box[3]/2)),int(box[0]-(box[2]/2)),int(box[1]+(box[3]/2)),int(box[0]+(box[2]/2))]
        return np.array(box_pixel)

    def box_normal_to_pixel3(self, box):
        # para trabajar con el multitraker de opencv usando detector de yolo
        box_pixel=[int(box[0]-(box[2]/2)),int(box[1]-(box[3]/2)),box[2],box[3]]
        return np.array(box_pixel)


    def get_localizationYolo(self,file, visual=False):

        boxes=[]
        scores=[]
        classes=[]

        #r = dn.detect(self.net, self.meta,file)
        image, r = self.image_detection(file, self.network, self.class_names, self.class_colors, .25)
        for i in range(0,len(r)):
        	boxes.append(np.asarray(r[i][2]))
        	scores.append(r[i][1])
        	classes.append(r[i][0])
        
        num_detections=len(r)
        boxes=np.squeeze(boxes)
        classes =np.squeeze(classes)
        scores = np.squeeze(scores)
        cls = classes.tolist()
        print(cls)
       
        idx_person = [i for i, v in enumerate(cls) if ((v=="person") and (float(scores[i])>0.2))]
        idx_withMask = [i for i, v in enumerate(cls) if ((v=="withMask") and (float(scores[i])>0.2))]
        idx_withoutMask = [i for i, v in enumerate(cls) if ((v=="withoutMask") and (float(scores[i])>0.2))]
        

        if len(idx_person) ==0:
             print('no person!')
        else:
              tmp_person_boxes=[]
              for idx in idx_person:
                  dim = image.shape[0:2]
                  if self.withKalman:
                      box = self.box_normal_to_pixel2(boxes[idx])
                      box_h = box[2] - box[0]
                      box_w = box[3] - box[1]
                  ratio = box_h/(box_w + 0.01)
                  tmp_person_boxes.append(box)
                  print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                  self.person_boxes = tmp_person_boxes

        if len(idx_withMask) ==0:
             print('no mask!')
        else:
              tmp_withMask_boxes=[]
              for idx in idx_withMask:
                  dim = image.shape[0:2]
                  if self.withKalman:
                      box = self.box_normal_to_pixel2(boxes[idx])
                      box_h = box[2] - box[0]
                      box_w = box[3] - box[1]
                  ratio = box_h/(box_w + 0.01)
                  tmp_withMask_boxes.append(box)
                  print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                  self.withMask_boxes = tmp_withMask_boxes


        if len(idx_withoutMask) ==0:
             print('no mask!')
        else:
              tmp_withoutMask_boxes=[]
              for idx in idx_withoutMask:
                  dim = image.shape[0:2]
                  if self.withKalman:
                      box = self.box_normal_to_pixel2(boxes[idx])
                      box_h = box[2] - box[0]
                      box_w = box[3] - box[1]
                  ratio = box_h/(box_w + 0.01)
                  tmp_withoutMask_boxes.append(box)
                  print(box, ', confidence: ', scores[idx], 'ratio:', ratio)
                  self.withoutMask_boxes = tmp_withoutMask_boxes

        return image, self.person_boxes, self.withMask_boxes, self.withoutMask_boxes
