"""
Caroline Li
CSC 391 Project 2: Local Feature Extraction, Detection, and Matching
2/4/2019

This script uses the webcam to capture video, extracts edges and corners from each frame 
and visualizes these features on each frame. 

"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,75,125)

    # Get corners
    dst = cv2.cornerHarris(gray,2,3,0.08)
    dst = cv2.dilate(dst,None)

    # Superimpose
    frame[edges>0] = [255,255,255]
    frame[dst>0.01*dst.max()] = [0,0,255]
    

    # Display the resulting frames
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()