import cv2
import numpy as np

cap = cv2.VideoCapture("General_public_preview.mp4")

if (cap.isOpened()==False):
    print('Error opening the file')

(sucess, image) = cap.read()

while sucess:
    image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    cv2.imshow('Results', image[:,:,::-1])

    key = cv2.waitKey(1) & 0xFF
    if key==ord("q"):
        break
    (sucess, image) =  cap.read()