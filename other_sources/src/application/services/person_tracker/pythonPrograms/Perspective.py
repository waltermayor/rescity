import numpy as np
import cv2
import simplejson
#from moviepy.editor import VideoFileClip
import os
import yaml

class Transform_Perspective():

    def __init__(self,image,pathVideo):
        self.pts=[]
        self.image=image
        self.Mtrasnf=[]
        self.maxHeight=0
        self.maxWidth=0
        self.width=0
        self.height=0
	self.cameraid=pathVideo

    def order_points(self,pts):

    	rect = np.zeros((4, 2), dtype = "float32")
    	# the top-left point will have the smallest sum, whereas
    	# the bottom-right point will have the largest sum
    	s = pts.sum(axis = 1)
    	rect[0] = pts[np.argmin(s)]
    	rect[2] = pts[np.argmax(s)]

    	# now, compute the difference between the points, the
    	# top-right point will have the smallest difference,
    	# whereas the bottom-left will have the largest difference
    	diff = np.diff(pts, axis = 1)
    	rect[1] = pts[np.argmin(diff)]
    	rect[3] = pts[np.argmax(diff)]

    	# return the ordered coordinates
    	return rect

    def four_point_transform(self,image, pts):

    	rect = self.order_points(pts)
    	(tl, tr, br, bl) = rect

    	# compute the width of the new image, which will be the
    	# maximum distance between bottom-right and bottom-left
    	# x-coordiates or the top-right and top-left x-coordinates
    	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    	maxWidth = max(int(widthA), int(widthB))

    	# compute the height of the new image, which will be the
    	# maximum distance between the top-right and bottom-right
    	# y-coordinates or the top-left and bottom-left y-coordinates
    	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    	maxHeight = max(int(heightA), int(heightB))

    	# now that we have the dimensions of the new image, construct
    	# the set of destination points to obtain a "birds eye view",
    	# (i.e. top-down view) of the image, again specifying points
    	# in the top-left, top-right, bottom-right, and bottom-left
    	# order
    	dst1 = np.array([
    		[0, 0],
    		[(maxWidth - 1)/2, 0],
    		[(maxWidth - 1)/2, (maxHeight - 1)/2],
    		[0, (maxHeight - 1)/2]], dtype = "float32")

        height, width = image.shape[:2]

        file=open("perpertiva.txt","a")
        file.write("-------------------------------------\n")

    	# compute the perspective transform matrix and then apply it
    	M = cv2.getPerspectiveTransform(rect, dst1)
        file.write("perspective 1\n")
        np.savetxt(file, np.vstack(M), fmt='%1.10f')
        #M2 = np.float32([[1,0,200],[0,1,200]])
        #image = cv2.warpAffine(image,M2,(400, 400))
        Ht = np.array([[1,0,height],[0,1,width],[0,0,1]]) # translate
        file.write("traslate \n")
        np.savetxt(file, np.vstack(Ht), fmt='%1.10f')
        file.write("perspective-translated \n")
        np.savetxt(file, np.vstack(Ht.dot(M)), fmt='%1.10f')
        warped = cv2.warpPerspective(image, Ht.dot(M), (width,height+maxHeight))

    	#warped = cv2.warpPerspective(image,M,(400, 400))
        file.close()

        # height, width = image.shape[:2]
        # warped = cv2.warpPerspective(image, M,(width, height))

    	return image, warped

    def point_transform_video(self,image, pts):

    	rect = self.order_points(pts)
    	(tl, tr, br, bl) = rect

    	# compute the width of the new image, which will be the
    	# maximum distance between bottom-right and bottom-left
    	# x-coordiates or the top-right and top-left x-coordinates
    	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    	maxWidth = max(int(widthA), int(widthB))

    	# compute the height of the new image, which will be the
    	# maximum distance between the top-right and bottom-right
    	# y-coordinates or the top-left and bottom-left y-coordinates
    	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    	maxHeight = max(int(heightA), int(heightB))

        self.maxHeight=maxHeight
        self.maxWidth=maxWidth

    	# now that we have the dimensions of the new image, construct
    	# the set of destination points to obtain a "birds eye view",
    	# (i.e. top-down view) of the image, again specifying points
    	# in the top-left, top-right, bottom-right, and bottom-left
    	# order
    	dst1 = np.array([
    		[0, 0],
    		[(maxWidth - 1)/2, 0],
    		[(maxWidth - 1)/2, (maxHeight - 1)/2],
    		[0, (maxHeight - 1)/2]], dtype = "float32")

        height, width = image.shape[:2]
        self.width=width
        self.height=height

        file=open("perpertiva.txt","a")
        file.write("-------------------------------------\n")

    	# compute the perspective transform matrix and then apply it
    	M = cv2.getPerspectiveTransform(rect, dst1)
        file.write("perspective 1\n")
        np.savetxt(file, np.vstack(M), fmt='%1.10f')
        #M2 = np.float32([[1,0,200],[0,1,200]])
        #image = cv2.warpAffine(image,M2,(400, 400))
        Ht = np.array([[1,0,height],[0,1,width],[0,0,1]]) # translate
        file.write("traslate \n")
        np.savetxt(file, np.vstack(Ht), fmt='%1.10f')
        file.write("perspective-translated \n")
        np.savetxt(file, np.vstack(Ht.dot(M)), fmt='%1.10f')
        self.Mtrasnf=Ht.dot(M)
    	#warped = cv2.warpPerspective(image,M,(400, 400))
        file.close()
        #warped = cv2.warpPerspective(image, Ht.dot(M), (width,height+maxHeight))
        # height, width = image.shape[:2]
        # warped = cv2.warpPerspective(image, M,(width, height))
        return self.Mtrasnf

    def warpPerspectiveTranslate(self):
        image=self.image
        warped = cv2.warpPerspective(image, self.Mtrasnf, (self.width,self.height+self.maxHeight))
        return warped

    def draw_points_callback(self,event,x,y,flags,param):

        if len(self.pts)<4:
            if event==cv2.EVENT_LBUTTONDOWN:
                self.pts.append((x,y))
                cv2.circle(self.image, (x, y), 7, [255], -1)
        #else:
            #print "ya selecciono los 4 putos"

    def getTransformationMatrix(self):

	outputFile=self.cameraid+".yaml"
        path='/detection/darknet/Program/cameraConfig/'+outputFile

    	if not os.path.exists(path):
            image = self.image
            cv2.namedWindow("selection points")
            cv2.setMouseCallback('selection points',self.draw_points_callback)
            while(1):
                #print height,"  ",width
                cv2.imshow('selection points',image)
                k = cv2.waitKey(1) & 0xFF
                if (k == 113):  # q is pressed
                  break

            pts = np.array(self.pts, dtype = "float32")
            #image, warped = transform.four_point_transform(image, pts)
            self.point_transform_video(image, pts)
            warped=self.warpPerspectiveTranslate()

            # clip1 = VideoFileClip("videosDePrueba/video5.mp4")#.subclip(4,49) # The first 8 seconds doesn't have any cars...
            # clip = clip1.fl_image(transform.warpPerspectiveTranslate)
            # clip.write_videofile('pruebas/perpectiva.mp4', audio=False)


            # show the original and warped images
            cv2.imshow("Original", self.image)
            cv2.imshow("Warped", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    	else:
	    print outputFile
    	    with open(path, 'r') as stream:
               data_loaded = yaml.load(stream)
               for j in range(0,len(data_loaded)):
                  cameraTrans=self.cameraid
                  self.Mtrasnf= data_loaded[cameraTrans]
                  #self.Mtrasnf=data["direction"]


        return self.Mtrasnf


if __name__ == '__main__':
    image = cv2.imread("image.jpg")
    transform=Transform_Perspective(image)
    transform.getTransformationMatrix()
