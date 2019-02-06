"""
Caroline Li
CSC 391 Project 1: Spatial and Frequency Filtering
2/4/2019

This script applies the 2D DFT to image data and displays the resulting coefficients in the 
Fourier doman in three different ways: with a 3D plot, 2D plot, and 2D plot of the log of 
the magnitudes. This script saves all figures as PNG images.

"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import skimage as ski

file = 'DSC_9259'   # name of file without extension

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
    fig.savefig(file + '_3d.jpg')

    # plot where y = 0
    # plt.plot(X, np.fft.fftshift(np.abs(f2)))
    # plt.xlim(-40, 40)
    # plt.show()

def plot2d(f2):
    """
    Visualizes the magnitude of the Fourier coefficients from the 2D 
    DFT of an image as a 2D plot.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
    """

    magnitude_img = np.fft.fftshift(np.abs(f2))
    magnitude_img = magnitude_img / magnitude_img.max()
    magnitude_img = ski.img_as_ubyte(magnitude_img)

    cv2.imshow('Magnitude Plot', magnitude_img)
    cv2.waitKey(0)

    cv2.imwrite(file + '_2d.jpg', magnitude_img)

def plotlog(f2):
    """
    Visualizes the log of the magnitude of the Fourier coefficients 
    from the 2D DFT of an image as a 2D plot.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
    """
    magnitude_img = np.fft.fftshift(np.log(np.abs(f2)))
    magnitude_img = magnitude_img / magnitude_img.max()
    magnitude_img = ski.img_as_ubyte(magnitude_img)

    cv2.imshow('Magnitude Plot', magnitude_img)
    cv2.waitKey(0)

    cv2.imwrite(file + '_log.jpg', magnitude_img)

def plotlog3d(f2, img):
    """
    Visualizes the log of the magnitude of the Fourier coefficients 
    from the 2D DFT of an image as a 3D plot.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
        img (ndarray, uint8): the original image
    """

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    Y = (np.linspace(-int(img.shape[0] / 2), int(img.shape[0] / 2) - 1, img.shape[0]))
    X = (np.linspace(-int(img.shape[1] / 2), int(img.shape[1] / 2) - 1, img.shape[1]))
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, np.fft.fftshift(np.log(np.abs(f2))), cmap=plt.get_cmap('coolwarm'), antialiased=False)

    plt.show()
    fig.savefig(file + '_log3d.jpg')

def zerolow(f2, img):
    """
    Takes an image and its 2D DFT and zeroes out the lowest 0.1% of 
    coefficients and displays the results.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
        img (ndarray, uint8): the original image
    """

    # get lowest 0.1% of coefficients and zero out
    low_values_flags = np.abs(f2) < ((f2.max() - f2.min()) / 1000)
    f2[low_values_flags] = 0

    plot3d(f2, img)

    # do inverse 2D DFT
    new_img = np.fft.ifft2(f2)
    new_img = new_img / new_img.max()

    greater_flag = new_img > 1
    new_img[greater_flag] = 1
    lesser_flag = new_img < -1
    new_img[lesser_flag] = -1

    new_img = ski.img_as_ubyte(new_img.real)

    cv2.imshow('Original Image', img)
    cv2.imshow('New Image', new_img)
    cv2.waitKey(0)
    
def zerohigh(f2, img):
    """
    Takes an image and its 2D DFT and zeroes out the highest 0.1% of 
    coefficients and displays the results.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
        img (ndarray, uint8): the original image
    """

    # get highest 0.1% of coefficients and zero out
    high_values_flags = np.abs(f2) > (f2.max() - ((f2.max() - f2.min()) / 1000))
    f2[high_values_flags] = 0

    # do inverse 2D DFT
    new_img = np.fft.ifft2(f2)
    new_img = new_img / new_img.max()

    greater_flag = new_img > 1
    new_img[greater_flag] = 1
    lesser_flag = new_img < -1
    new_img[lesser_flag] = -1

    new_img = ski.img_as_ubyte(new_img.real)

    cv2.imshow('Original Image', img)
    cv2.imshow('New Image', new_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'Images/' + file + '.JPG') # change name of file accordingly

    if not os.path.exists(filename):
        print("File does not exist.")
        exit()

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # take 2D DFT
    f2 = np.fft.fft2(img.astype(float))

    # visualize as 3D plot
    plot3d(f2, img)

    # visualize as 2D image
    plot2d(f2)

    # plot log of magnitude + 1
    plotlog(f2)

    # plot log of magnitude + 1 in 3D
    plotlog3d(f2, img)
    