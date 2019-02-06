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
    fig.savefig('dft_3d.png')

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

    cv2.imwrite('dft_2d.png', magnitude_img)

def plotlog(f2, img):
    """
    Visualizes the log of the magnitude of the Fourier coefficients 
    from the 2D DFT of an image as a 2D plot.

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
    fig.savefig('dft_log.png')

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'Images/DSC_9259.JPG') # change name of file accordingly

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
    plotlog(f2, img)