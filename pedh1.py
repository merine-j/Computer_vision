import cv2    # Import the OpenCV library to enable computer vision
import numpy as np   # Import the NumPy numerical computing library
import pygame        #import pygame for audio alert
from imutils.object_detection import non_max_suppression  # Handle overlapping

#initializes the pygame module
pygame.mixer.init()

def ped():

    ped_ccd=cv2.HOGDescriptor()
    ped_ccd.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    vid=cv2.VideoCapture(r'C:\Computer vision\projects\Untitled video - Made with Clipchamp (6).mp4')

    pygame.mixer.music.load(r'C:\Computer vision\projects\sfx_seatbelt-warning_auto-80273.mp3')

    while True:
        suc,frame=vid.read()
        frame=cv2.resize(frame,(0,0),fx=0.8,fy=0.8)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        pedestrians,_=ped_ccd.detectMultiScale(gray,winStride=(8,8),padding=(10,10),scale=1.1,hitThreshold=0.3)
        print(len(pedestrians))

        for x,y,w,h in pedestrians:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

        pedestrians=np.array([[x,y,x+w,y+h]for (x,y,w,h) in pedestrians],dtype=int)
        selection=non_max_suppression(pedestrians,probs=None,overlapThresh=0.45)
        
        for x1,y1,x2,y2 in selection:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    
            cv2.putText(frame,'Pedestrian ahead...Go slow..',(15,55),cv2.FONT_HERSHEY_DUPLEX,0.9,(0,0,255),2)
        cv2.putText(frame,'Pedestrian alert system',(15,25),cv2.FONT_HERSHEY_COMPLEX,0.9,(0,0,),2)
                
        if len(selection)>0:
            pygame.mixer.music.play()

        cv2.imshow('video',frame)
        
        if cv2.waitKey(1)&0XFF==ord('q'):
            break
    
    vid.release()
    
    cv2.destroyAllWindows()

ped()