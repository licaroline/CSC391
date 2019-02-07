"""
Caroline Li
CSC 391 Project 1: Spatial and Frequency Filtering
2/4/2019

This script applies a spatial filter of user-defined dimension k (k-by-k filter) to an image
and displays the original image, the filter, and the filtered image. This script then saves
the filtered image to a JPG file called filtered_image.JPG.

"""

import os
import cv2
import numpy as np

def box(img, k):
    """
    Filters an image using a k-by-k box filter.

    Args:
        img (ndarray, uint8): image to be filtered
        k (int): dimensions for the filter
    """

    # create the box filter
    box_filter = np.ones((k, k), np.float32)
    box_filter = box_filter / box_filter.sum()

    print("Box Filter: ")
    print(box_filter)

    # create space for filtered image
    new_img = np.zeros(img.shape, np.uint8)

    # fill in values for new image
    new_img[:, :, 0] = cv2.filter2D(img[:, :, 0], -1, box_filter)
    new_img[:, :, 1] = cv2.filter2D(img[:, :, 1], -1, box_filter)
    new_img[:, :, 2] = cv2.filter2D(img[:, :, 2], -1, box_filter)

    return new_img

if __name__ == "__main__":
    DIR = os.path.dirname(__file__)
    FILENAME = os.path.join(DIR, 'Images/DSC_9259.JPG') # change name of file accordingly

    if not os.path.exists(FILENAME):
        print("File does not exist.")
        exit()

    IMG = cv2.imread(FILENAME, 1)

    # show original image
    cv2.imshow('original image', IMG)
    cv2.waitKey(0)

    # get size of filter
    print("Enter k for the dimensions of the k-by-k filter: ")
    K = int(input())
    print(f"Filter is of dimension {k}x{k}")

    # filter image
    FILTERED_IMG = box(IMG, K)

    # show filtered image
    cv2.imshow('filtered image', FILTERED_IMG)
    cv2.waitKey(0)

    # save filtered image
    cv2.imwrite(os.path.join(dir, 'filtered_image.JPG'), FILTERED_IMG)
