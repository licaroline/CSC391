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

file = 'DSC_9259-0.50'   # name of file without extension

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
    magnitude_img = np.fft.fftshift(np.log(np.abs(f2) + 1))
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

    ax.plot_surface(X, Y, np.fft.fftshift(np.log(np.abs(f2) + 1)), cmap=plt.get_cmap('coolwarm'), antialiased=False)

    plt.show()
    fig.savefig(file + '_log3d.jpg')

def zerolow(f2, img):
    """
    Takes an image and its 2D DFT and zeroes out the lowest 
    coefficients and displays the results.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
        img (ndarray, uint8): the original image
    """

    # get mid-point of image
    w, h = img.shape
    half_w, half_h = int(w / 2), int(h / 2)

    n = 50  # half of width of box we're cutting out

    f2 = np.fft.fftshift(f2)
    f2[half_w - n : half_w + n + 1, half_h - n : half_h + n + 1] = 0
    f2 = np.fft.ifftshift(f2)

    plot3d(f2, img)
    showimg(f2)
    plotlog(f2)

def zerohigh(f2, img):
    """
    Takes an image and its 2D DFT and zeroes out the highest 
    coefficients and displays the results.

    Args:
        f2 (ndarray, complex128): 2D DFT of an image
        img (ndarray, uint8): the original image
    """

    w, h = img.shape
    half_w, half_h = int((w - 100) / 2), int((h - 100) / 2)

    f2 = np.fft.fftshift(f2)
    f2[0:half_w, :] = 0
    f2[w - half_w - 1: w, :] = 0
    f2[:, h - half_h - 1 : h] = 0
    f2[:, 0:half_h] = 0
    f2 = np.fft.ifftshift(f2)

    plot3d(f2, img)
    showimg(f2)
    plotlog(f2)

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

    cv2.imwrite('img.JPG', img)

if __name__ == "__main__":
    DIR = os.path.dirname(__file__)
    FILENAME = os.path.join(DIR, 'Images/' + file + '.JPG')

    if not os.path.exists(FILENAME):
        print("File does not exist.")
        exit()

    IMG = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)

    # take 2D DFT
    F2 = np.fft.fft2(IMG.astype(float))

    # visualize as 3D plot
    plot3d(F2, IMG)

    # visualize as 2D image
    plot2d(F2)

    # plot log of magnitude + 1
    plotlog(F2)

    # plot log of magnitude + 1 in 3D
    plotlog3d(F2, IMG)
    