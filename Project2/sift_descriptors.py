"""
Caroline Li
CSC 391 Project 2: Local Feature Extraction, Detection, and Matching
2/4/2019

This script takes an image and uses a SIFT descriptor to draw keypoints on the image.

"""

import cv2
import numpy as numpy

# Get image and make it grayscale
img = cv2.imread('penguins.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Create SIFT descriptor
sift = cv2.xfeatures2d.SIFT_create(0,3,0.1,10,1.6)
kp = sift.detect(gray,None)

# Apply to image
img = cv2.drawKeypoints(gray,kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Display image
cv2.imshow('SIFT keypoints',img)
cv2.waitKey(0)

# cv2.imwrite('penguins_sift.jpg', img)
