import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import *
cmap = 'gray'

# load data, grayscale then binary, crop
img_gray = cv2.imread('data/test_3.png', cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img = crop_borders(im_bw)

# opening (erosion maybe?) dilate/erose and open/close function name seems to be flipped
erosion_size = 4
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
# img_eroded = cv2.dilate(img, kernel) #keeping this just in case
img_open = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# connected component
ret, labels = cv2.connectedComponents(img_open)

# visualization
images = [img, img_open]
subplot_rows = 1
subplot_cols = len(images)
for index, image in enumerate(images):
	fig = plt.subplot(subplot_rows,subplot_cols,index+1)
	plt.imshow(image, cmap)
	fig.axes.get_yaxis().set_ticks([])
	fig.axes.get_xaxis().set_ticks([])
plt.show()
