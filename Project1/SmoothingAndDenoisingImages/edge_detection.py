import os
import cv2
from matplotlib import pyplot as plt

# get noisy and non-noisy puppy image
dir = os.path.abspath('..')
filename = os.path.join(dir, 'Images/DSC_9259.JPG')

if not os.path.exists(filename):
    print("File does not exist.")
    exit()

img = cv2.imread(filename, 1)

filename = os.path.join(dir, 'Images/DSC_9259-0.50.JPG')

if not os.path.exists(filename):
    print("File does not exist.")
    exit()

noisy_img = cv2.imread(filename, 1)

# use Canny Edge Detection
edges = cv2.Canny(img, 100, 200)
noisy_edges = cv2.Canny(noisy_img, 100, 200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

plt.subplot(121),plt.imshow(noisy_img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(noisy_edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()