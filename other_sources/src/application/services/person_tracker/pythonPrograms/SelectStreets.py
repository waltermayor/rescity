#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#from Tkinter import *
#from PIL import Image
#from PIL import ImageTk
import cv2
import numpy as np
import structures
import yaml
import os

class Sel_Streets():

    def __init__(self,image,pathVideo,matrixTransf):
        self.rois=[]
        self.roi=[]
        self.color=(np.random.randint(64, 255),np.random.randint(64, 255), np.random.randint(64, 255))
        self.im=image
        self.imgr=image
        self.masks=[]
        self.streetList=[]
        self.maskse=np.zeros_like(self.im)  # todas las calles se van sumando
        self.masklo=np.zeros_like(cv2.cvtColor(self.imgr, cv2.COLOR_BGR2GRAY)) # una sola calle a la vez
        self.intersections=[]
        #self.masklo = np.zeros(self.imgr.shape, np.uint8)
        self.colorlist=[20,20,30,40,50,60,70]
        self.i=0
        self.cameraid=pathVideo
	self.matrixTransf=matrixTransf


    def draw_Roi_callback(self,event,x,y,flags,param):

        if event==cv2.EVENT_LBUTTONDOWN:
            self.roi.append((x,y))

        elif event==cv2.EVENT_LBUTTONUP:
            if len(self.roi)>1:
                cv2.line(self.im,(self.roi[-2][0],self.roi[-2][1]),(x,y),self.color,3)

        elif event==cv2.EVENT_RBUTTONDOWN:
            self.masklo=np.zeros_like(cv2.cvtColor(self.imgr, cv2.COLOR_BGR2GRAY))

            # marcara con todas las calles
            cv2.fillPoly(self.maskse, [np.array(self.roi)],self.color)
            #cv2.imshow('selection',self.maskse)
            #-----------------------------------

            # mascara localizacion de cada calle
            cv2.fillPoly(self.masklo, [np.array(self.roi)],[self.colorlist[self.i]])
            #cv2.imshow('location',self.masklo)

            # guardar caracteristicas de la calle
            street=structures.Street(self.i,self.masklo)
            streetDirection = raw_input("Digite la letra que define la dirección de la calle (H)-horizotal (V)-vetical: ")
            street.direction=streetDirection
	    wayDirection = raw_input("Digite el sentido de la calle hacia: (U)-arriba (D)-abajo (UD)-ambas (L)-izquierda (R)-derecha (LR)-ambas")
            street.wayDirection=wayDirection

            self.masks.append(self.masklo)
            self.rois.append(self.roi)
            self.streetList.append(street)

            self.roi=[]
            self.i=self.i+1

            self.color=(np.random.randint(64, 255), np.random.randint(64, 255), np.random.randint(64, 255))
            print len(self.rois)

    def find_intersections(self):
        instersection=np.zeros_like(cv2.cvtColor(self.imgr, cv2.COLOR_BGR2GRAY))
        for idx1 in range(0,len(self.streetList)-1):
            for idx2 in range(idx1+1,len(self.streetList)):
                instersection2=self.streetList[idx1].mask*self.streetList[idx2].mask
                instersection=instersection+instersection2
                edged = cv2.Canny(instersection2, 5,80)
                _, contours, _=cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours)>0:
                    M = cv2.moments(contours[0])
                    print M,"id2 ",idx2,"id1 ",idx1
                    if M["m00"]!=0:
                        cX= int(M["m10"] / M["m00"])
                        cY= int(M["m01"] / M["m00"])
                        self.streetList[idx1].interseCentroids.append([cX,cY])
                        self.streetList[idx2].interseCentroids.append([cX,cY])
                        cv2.circle(instersection, (cX, cY), 7, [255], -1)

			centroidTrans=self.matrixTransf.dot(np.expand_dims(np.array([cX,cY,1]), axis=0).T)
           		centroidTrans2D = centroidTrans / np.tile(centroidTrans[-1, :], (3, 1)) # Divide todo por el valor último del vector.
            		centroidTrans2D=centroidTrans2D[:2,0]
		
			self.streetList[idx1].interseCentroidsTransf.append([centroidTrans2D[0],centroidTrans2D[1]])
                        self.streetList[idx2].interseCentroidsTransf.append([centroidTrans2D[0],centroidTrans2D[1]])
                    else:
                        print "the interception area is too small"
                else:
                    area=0
        #cv2.imshow('intersection',instersection)


    def streets_rois(self):

     	outputFile=self.cameraid+".yaml"
        path='/detection/darknet/Program/cameraConfig/'+outputFile
        print path
        if not os.path.exists(path):
            cv2.namedWindow("selection streets")
            #cv2.namedWindow("selection")
            #cv2.namedWindow("location")
            #cv2.namedWindow("intersection")
            cv2.setMouseCallback('selection streets',self.draw_Roi_callback)
            while(1):
                cv2.imshow('selection streets',self.im)
                k = cv2.waitKey(1) & 0xFF
                if (k == 113):  # q is pressed
                  break

            self.find_intersections()
            stream= open(outputFile, 'w')
	    cameraTrans=self.cameraid
	    new_data={cameraTrans:self.matrixTransf}
	    yaml.dump(new_data,stream)
            new_data={}
            for j in range (0,self.i):
                data=self.streetList[j]
                box="street"+str(j)
                new_data = {box: {'id': data.id, 'mask':data.mask ,'intersections':data.intersections,
                'interseCentroids': data.interseCentroids , 'direction': data.direction, 'wayDirection':data.wayDirection}}
                yaml.dump(new_data,stream)

            # while(1):
            #     self.find_intersections()
            #     k = cv2.waitKey(1) & 0xFF
            #     if (k == 113):  # q is pressed
            #       break
            cv2.destroyAllWindows()
        else:
            print "esa camara ya fue configurada"
            with open(path, 'r') as stream:
                data_loaded = yaml.load(stream)
            for j in range(0,len(data_loaded)-1):
                box="street"+str(j)
                data= data_loaded[box]
                street=structures.Street(data["id"],data["mask"])
                street.direction=data["direction"]
                street.intersections=data["intersections"]
                street.interseCentroids=data["interseCentroids"]
		street.wayDirection=data["wayDirection"]
		for idx,cent in enumerate(street.interseCentroids):

		    centroidTrans=self.matrixTransf.dot(np.expand_dims(np.array([cent[0],cent[1],1]), axis=0).T)
                    centroidTrans2D = centroidTrans / np.tile(centroidTrans[-1, :], (3, 1)) # Divide todo por el valor último del vector.
                    centroidTrans2D=centroidTrans2D[:2,0]

                    street.interseCentroidsTransf.append([centroidTrans2D[0],centroidTrans2D[1]])

		#street.interseCentroidsTransf=data["interseCentroidsTransf"]
                self.streetList.append(street)




if __name__ == "__main__":

    im = cv2.imread("image.jpg")
    rois=Sel_Streets(im)
    rois.streets_rois()
