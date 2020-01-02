# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:59:57 2019

@author: Sonu
"""

#import numpy as np
import cv2 

face_cascade = cv2.CascadeClassifier(r'C:\Users\Sonu\Documents\Carrear\HACKATHONS\Computer Vision\Harr Cascacde\Frontal_Face.xml')

image=cv2.imread(r'C:\Users\Sonu\Documents\Carrear\HACKATHONS\Computer Vision\Counting Faces\image_data\10008.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grayImage, scaleFactor=1.2, minNeighbors=5)
print(type(faces))
print(faces.shape)
print("Number of faces detected: " + str(faces.shape[0]))

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
    
cv2.rectangle(image, ((0,image.shape[0] -25)),(270, image.shape[0]), (255,255,255), -1)
cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)    
cv2.imshow('Image with faces',image)
cv2.waitKey(2000)
cv2.destroyAllWindows()