from person_tracker import PersonTracker
import sys, os
from moviepy.editor import VideoFileClip, concatenate, ImageClip, ImageSequenceClip 
import glob
from natsort import natsorted
import base64
from io import BytesIO
from PIL import Image


if __name__ == "__main__":
    clips =[]
    obj = PersonTracker()
    #obj.beginAnalisisVideo(sys.argv[1])
    voutput = '/app/service/src/application/services/person_tracker/videoOutputs/o' + "video1" + '.mp4'
    IMAGE_DIRECTORY = "/app/service/src/application/services/person_tracker/videoInputs/camara2/"
    directory= natsorted(os.listdir(IMAGE_DIRECTORY))

    """ for file in directory:
        if file.startswith('camara'):
            print(file)
            print(os.path.join(IMAGE_DIRECTORY, file))
            img = obj.analyseImage(os.path.join(IMAGE_DIRECTORY, file))
            clips.append(img) """
    
    for file in directory:
        if file.startswith('camara'):
            with open(os.path.join(IMAGE_DIRECTORY, file), "rb") as f:
                im_b64 = base64.b64encode(f.read())
            print("##############################################")
            print(im_b64)
            print("##############################################")
            img = obj.beginAnalisisImage(im_b64)
            clips.append(img)

    clip = ImageSequenceClip(clips, fps=24)
    clip.write_videofile(voutput)
            
    
