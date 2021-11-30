#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kyleguan
@updated: walter
"""
import numpy as np
import cv2

class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);



def box_iou2(a, b):
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

    return float(s_intsec)/(s_a + s_b -s_intsec)



def convert_to_pixel(box_yolo, img, crop_range):
    '''
    Helper function to convert (scaled) coordinates of a bounding box
    to pixel coordinates.

    Example (0.89361443264143803, 0.4880486045564924, 0.23544462956491041,
    0.36866588651069609)

    crop_range: specifies the part of image to be cropped
    '''

    box = box_yolo
    imgcv = img
    [xmin, xmax] = crop_range[0]
    [ymin, ymax] = crop_range[1]
    h, w, _ = imgcv.shape

    # Calculate left, top, width, and height of the bounding box
    left = int((box.x - box.w/2.)*(xmax - xmin) + xmin)
    top = int((box.y - box.h/2.)*(ymax - ymin) + ymin)

    width = int(box.w*(xmax - xmin))
    height = int(box.h*(ymax - ymin))

    # Deal with corner cases
    if left  < 0    :  left = 0
    if top   < 0    :   top = 0

    # Return the coordinates (in the unit of the pixels)

    box_pixel = np.array([left, top, width, height])
    return box_pixel

def convert_to_cv2bbox(bbox, img_dim = (1280, 720)):
    '''
    Helper fucntion for converting bbox to bbox_cv2
    bbox = [left, top, width, height]
    bbox_cv2 = [left, top, right, bottom]
    img_dim: dimension of the image, img_dim[0]<-> x
    img_dim[1]<-> y
    '''
    left = np.maximum(0, bbox[0])
    top = np.maximum(0, bbox[1])
    right = np.minimum(img_dim[0], bbox[0] + bbox[2])
    bottom = np.minimum(img_dim[1], bbox[1] + bbox[3])

    return (left, top, right, bottom)

def draw_line_future(img,point1,point2,color):
    cv2.line(img,(int(point1[0]),int(point1[1])),(int(point2[0]),int(point2[1])), 255, 2)
    return img

def draw_line_traker(img,centroids,color):
    cen1=centroids[0]
    cen2=centroids[len(centroids)-1]
    print(cen1, cen2)
    cv2.line(img,(cen1[0],cen1[1]),(cen2[0],cen2[1]), color, 4, 8, 0)
    return img

def draw_poliLine_traker(img,centroids,color):
    points=np.array(centroids).reshape((-1,1,2)).astype(np.int32)
    #cv2.drawContours(img,[points],-1,color,3)
    cv2.polylines(img, [points], False, color, 1);
    return img

def draw_centroids(img,trk):
    centroid = (trk.centroids[-1][0],trk.centroids[-1][1])
    cv2.circle(img, centroid, 3, trk.colorProb, -1)
    return img
#def draw_box_label(img, bbox_cv2, box_color=(0, 255, 255), show_label=True):
def draw_box_label(img, bbox_cv2, box_color,id,box_colorP, number_person, without_mask, show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]

    # Draw the bounding box
    overlay=img.copy()
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 1)
    cv2.rectangle(overlay, (left, top), (right, bottom), box_colorP,-1,1) # representa la probabilidad de choque
    alpha=0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img);

    if show_label:
        cv2.rectangle(img, (10, 10), (60,40), (100,100,100), -1, 1)
        cv2.putText(img,str(number_person),(15,35), font, 0.7, font_color, 1, cv2.LINE_AA)

        cv2.rectangle(img, (70, 10), (120,40), (100,100,100), -1, 1)
        cv2.putText(img,str(without_mask),(75,35), font, 0.7, font_color, 1, cv2.LINE_AA)

        cv2.rectangle(img, (left-2, top-20), (right+2, top), box_color, -1, 1)
        cv2.putText(img,str(id),(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)


    return img
