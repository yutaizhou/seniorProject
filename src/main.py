import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import *
cmap = 'gray'

# load data, grayscale then binary, crop
img_gray = cv2.imread('data/test_1.png', cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img = crop_borders(im_bw)

# erosion
erosion_type = cv2.MORPH_ELLIPSE
erosion_size = 4
kernel = cv2.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
img_eroded = cv2.dilate(img, kernel)

# opening
img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# visualization
subplot_rows = 1
subplot_cols = 3
images = [img, img_eroded, img_open]
for index, image in enumerate(images):
	fig = plt.subplot(subplot_rows,subplot_cols,index+1)
	plt.imshow(image, cmap)
	fig.axes.get_yaxis().set_ticks([])
	fig.axes.get_xaxis().set_ticks([])
plt.show()
