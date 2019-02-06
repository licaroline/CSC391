import os
import cv2

dir = os.path.abspath('..')
filename = os.path.join(dir, 'Images/DSC_9259-0.02.JPG')

if not os.path.exists(filename):
    print("File does not exist.")
    exit()

img = cv2.imread(filename, 1)

# gaussian filter 3x3
g3 = cv2.GaussianBlur(img,(3, 3), 0)
cv2.imwrite('0.02g3.JPG', g3)

# gaussian filter 9x9
g9 = cv2.GaussianBlur(img,(9, 9), 0)
cv2.imwrite('0.02g9.JPG', g9)

# gaussian filter 27x27
g27 = cv2.GaussianBlur(img,(27, 27), 0)
cv2.imwrite('0.02g27.JPG', g27)

# median filter 3x3
m3 = cv2.medianBlur(img, 3)
cv2.imwrite('0.02m3.JPG', m3)

# median filter 9x9
m9 = cv2.medianBlur(img, 9)
cv2.imwrite('0.02m9.JPG', m9)

# median filter 27x27
m27 = cv2.medianBlur(img, 27)
cv2.imwrite('0.02m27.JPG', m27)
