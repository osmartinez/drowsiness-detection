from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
from socket import *
import socket
import numpy as np
import sys
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import time
import math
from imutils import face_utils
from threading import Thread
import playsound
import subprocess
import os

class DetectorSueno():
    def __init__(self):
        self._shape_predictor = "models/shape_predictor_68_face_landmarks.dat"
        self._resolucion = (640, 480)
        self._calidad_imagen=[int(cv2.IMWRITE_JPEG_QUALITY),70]
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(self._shape_predictor)
        self._UMBRAL_EAR = 0.27
        self._MIN_N_FRAMES = 3
	def detecta_puntos_clave(self,imagePath,detector,predictor):
            image = cv2.imread(imagePath)
            image = imutils.resize(image,width=500)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            rects = self._detector(gray,1)

            for (i,rect) in enumerate(rects):
                    shape = self._predictor(gray,rect)
                    shape = face_utils.shape_to_np(shape)
                    (x,y,w,h) = face_utils.rect_to_bb(rect)
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(image, "Cara #{}".format(i + 1), (x - 10, y - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    for (x, y) in shape:
                            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            # Conversion de la imagen para mostrarla en el notebook
            img2 = image[:,:,::-1]
            cv2.imshow('Puntos clave',img2)
            cv2.waitKey(1)
    def reproduce_alarma(self,path):
        playsound.playsound(path)
        
    def euclidean_dist(self,p1,p2):
        return math.sqrt(((p1[0]-p2[0])*(p1[0]-p2[0]))+((p1[1]-p2[1])*(p1[1]-p2[1])))
    
    def eye_aspect_ratio(self,eye):
        A = self.euclidean_dist(eye[1],eye[5])#dist.euclidean(eye[1], eye[5])
        B = self.euclidean_dist(eye[2],eye[4])#dist.euclidean(eye[2], eye[4])
        C = self.euclidean_dist(eye[0],eye[3])#dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
	
    def run(self):
        self._cam= PiCamera()
        self._cam.resolution = (640, 480)
        self._cam.framerate = 60
        self._captura = PiRGBArray(self._cam, size=(640, 480))
        time.sleep(2)
        contador=0
        total=0
        sonido="audio/alarm.mp3"
        (lStart,lEnd) = (42,48)#face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart,rEnd) =(36,42) #face_utils.FACIAL_LANDMARKS_IDX["right_eye"]
                                                         
        for frame in self._cam.capture_continuous(self._captura, format="bgr", use_video_port=True):
            try:
                frame = frame.array
                #frame=frame[:,:,::-1]
                frame=imutils.resize(frame,width=250)
                
                #frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                
                #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                #gray2 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                #cv2.imshow('primera0',gray)
                #cv2.waitKey(1)
                rects = self._detector(frame, 0)
                for rect in rects:
                    shape = self._predictor(frame, rect)
                    shape = face_utils.shape_to_np(shape)
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    if ear < self._UMBRAL_EAR:
                        contador += 1
                        if contador>=self._MIN_N_FRAMES:
                            subprocess.call(["omxplayer",sonido],stdout=open(os.devnull,'wb'))
                            cv2.putText(frame, "Te duermes!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            contador=0
                    else:
                        contador=0
                        
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        #webcam_preview.set_data(frame)

                        #plt.draw()
                        #display.clear_output(wait=True)
                        #display.display(plt.gcf())
                        #plt.pause(0.1)
                cv2.imshow('capturadora',frame)
                cv2.waitKey(32)
                self._captura.truncate(0)
            except KeyboardInterrupt:
                self._cam.close()
                cv2.destroyAllWindows()
                break
                        
import sys

def error():
    print 'Error: Argumentos no validos!'
    print 'Modo de uso: python <modulo>'
    time.sleep(4)
    exit()

def main():
    args = sys.argv
    if len(args)>1:
        error()
    
    detector = DetectorSueno()
    detector.run()
	
if __name__ == "__main__": main()