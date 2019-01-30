import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from util import *
cmap = 'gray'

# load data, grayscale, crop
img_og = cv.imread('data/test_2.png')
img_gray = cv.cvtColor(img_og,cv.COLOR_BGR2GRAY)
img = crop_borders(img_gray)

# erosion
erosion_type = cv.MORPH_ELLIPSE
erosion_size = 4
kernel = cv.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
img_eroded = cv.dilate(img, kernel)

plt.subplot(1,2,1)
plt.imshow(img, cmap)
plt.subplot(1,2,2)
plt.imshow(img_eroded, cmap)
plt.show()
