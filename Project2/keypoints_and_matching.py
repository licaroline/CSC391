"""
Caroline Li
CSC 391 Project 2: Local Feature Extraction, Detection, and Matching
2/4/2019

This script takes two images and performs feature matching using Harris Corner
Detection or SIFT detectors.

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

def bf_matcher(img1, img2, kp1, kp2, des1, des2):
    """
    Applies feature matching using Brute-Force matching. Code taken from OpenCV Python's
    online guide for Feature Matching.

    Args:
        img1 (ndarray, uint8): query image
        img2 (ndarray, uint8): train image
        kp1 (list): keypoints from img1
        kp2 (list): keypoints from img2
        des1(ndarray, float32): descriptors from img1
        des2(ndarray, float32): descriptors from img2
    """

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img3

def use_harris(img1, img2):
    """
    Finds keypoints using Harris Corner Detection and descriptors using a SIFT
    detector.

    Args:
        img1 (ndarray, uint8): query image
        img2 (ndarray, uint8): train image
    """
    # Get corners with Harris Corner detection
    corners1 = cv2.cornerHarris(img1,2,3,0.08)
    corners1 = cv2.dilate(corners1,None)
    corners1 = corners1>0.01*corners1.max()     # true/false array corresponding with image for corners
    corners2 = cv2.cornerHarris(img2,2,3,0.08)
    corners2 = cv2.dilate(corners2,None)
    corners2 = corners2>0.01*corners2.max()     # true/false array corresponding with image for corners

    row1, col1 = corners1.shape
    row2, col2 = corners2.shape

    # Get keypoints from corners
    kp1 = []
    kp2 = []

    for y in range(row1):
        for x in range(col1):
            if corners1[y][x]:
                kp1.append(cv2.KeyPoint(x, y, 13))
    
    for y in range(row2):
        for x in range(col2):
            if corners2[y][x]:
                kp2.append(cv2.KeyPoint(x, y, 13))

    # Create SIFT descriptor
    sift = cv2.xfeatures2d.SIFT_create(0,3,0.1,10,1.6)

    # Get descriptors with SIFT
    des1 = [sift.compute(img1,[kp])[1][0] for kp in kp1]
    des2 = [sift.compute(img2,[kp])[1][0] for kp in kp2]

    des1 = np.array(des1, dtype='float32')
    des2 = np.array(des2, dtype='float32')

    img3 = bf_matcher(img1, img2, kp1, kp2, des1, des2)

    return img3
    
def use_sift(img1, img2):
    """
    Finds keypoints and descriptors with a SIFT detector.

    Args:
        img1 (ndarray, uint8): query image
        img2 (ndarray, uint8): train image
    """
    # Create SIFT descriptor
    sift = cv2.xfeatures2d.SIFT_create(0,3,0.1,10,1.6)

    # Get keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    img3 = bf_matcher(img1, img2, kp1, kp2, des1, des2)

    return img3

if __name__ == "__main__":
    img1 = cv2.imread('penguins.jpg',0)
    img2 = cv2.imread('matching5.jpg',0)
    img3 = use_harris(img1, img2)   # adjust for sift or harris

    plt.imshow(img3),plt.show()
    plt.imsave('matching5_harris.jpg', img3)