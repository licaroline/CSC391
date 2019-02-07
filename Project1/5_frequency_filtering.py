"""
Caroline Li
CSC 391 Project 1: Spatial and Frequency Filtering
2/4/2019

This script takes an image and applies a low-pass Butterworth filter and a high-pass
Butterworth filter. It displays a plot of the magnitudes before and after the filters,
which it saves. It also shows the images after filtering.

"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski
from scipy import signal
import psychopy.filters

file = 'DSC_9259-0.50'   # name of file without extension
cutoff_freq = 0.05  # cutoff frequency for Butterworth filters

def plot3d(f2, img):
    """
    Visualizes the magnitude of the Fourier coefficients from the 2D 
    DFT of an image as a 3D plot.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
        img (ndarray, uint8): the original image
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Y = (np.linspace(-int(img.shape[0] / 2), int(img.shape[0] / 2) - 1, img.shape[0]))
    X = (np.linspace(-int(img.shape[1] / 2), int(img.shape[1] / 2) - 1, img.shape[1]))
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, np.fft.fftshift(np.abs(f2)), cmap=plt.get_cmap('coolwarm'), antialiased=False)

    plt.show()
    print('Enter name for plot to be saved as:')
    name = input()
    fig.savefig(name + '.JPG')

def plotlog(f2):
    """
    Visualizes the log of the magnitude of the Fourier coefficients 
    from the 2D DFT of an image as a 2D plot.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
    """
    magnitude_img = np.fft.fftshift(np.log(np.abs(f2) + 1))
    magnitude_img = magnitude_img / magnitude_img.max()
    magnitude_img = ski.img_as_ubyte(magnitude_img)

    cv2.imshow('Magnitude Plot', magnitude_img)
    cv2.waitKey(0)

def showimg(f2):
    """
    Takes an image's 2D DFT and shows it as an image.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
    """
    # do inverse 2D DFT
    img = np.fft.ifft2(f2)
    img = img / img.max()

    # make sure everything's between -1 and 1
    greater_flag = img > 1
    img[greater_flag] = 1
    lesser_flag = img < -1
    img[lesser_flag] = -1

    img = ski.img_as_ubyte(img.real)

    cv2.imshow('Image', img)
    cv2.waitKey(0)

def lowbutter(f2, img):
    """
    Applies a low-pass Butterworth filter on the image.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
        img (ndarray, uint8): the original image
    """
    filt = psychopy.filters.butter2d_lp(size=img.shape, cutoff=cutoff_freq, n=2)

    filtered_img = np.fft.fftshift(f2) * filt
    filtered_img = np.fft.ifftshift(filtered_img)

    plotlog(filtered_img)
    showimg(filtered_img)
    plot3d(filtered_img, img)

def highbutter(f2, img):
    """
    Applies a high-pass Butterworth filter on the image. 

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
        img (ndarray, uint8): the original image
    """
    
    filt = psychopy.filters.butter2d_hp(size=img.shape, cutoff=cutoff_freq, n=2)

    filtered_img = np.fft.fftshift(f2) * filt
    filtered_img = np.fft.ifftshift(filtered_img)

    plotlog(filtered_img)
    showimg(filtered_img)
    plot3d(filtered_img, img)

if __name__ == "__main__":
    DIR = os.path.dirname(__file__)
    FILENAME = os.path.join(DIR, 'Images/' + file + '.JPG')

    if not os.path.exists(FILENAME):
        print("File does not exist.")
        exit()

    IMG = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)

    # take 2D DFT
    F2 = np.fft.fft2(IMG.astype(float))

    print('Plotting magnitudes of original Fourier coefficients...')
    plot3d(F2, IMG)

    print('Running a low-pass Butterworth filter on image...')
    lowbutter(F2, IMG)

    print('Running a high-pass Butterworth filter on image...')
    highbutter(F2, IMG)
