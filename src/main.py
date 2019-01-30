import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import *


# load data, grayscale then binary, crop, invert
img_gray = cv2.imread('data/test_1.png', cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bw_crop = crop_borders(im_bw)
img = img_bw_crop_inv = cv2.bitwise_not(img_bw_crop)


# opening
erosion_size = 3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# connected component
ret, labels = cv2.connectedComponents(img_open)

label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2HSV)
labeled_img[label_hue==0] = 0

# visualization
images = [('Input Grayscale', img_gray),
		  ('Post Processed', img),
		  ('Opening', img_open),
		  ('Connected Components', labeled_img)]
subplot_rows = 1
subplot_cols = len(images)
for index, title_and_image in enumerate(images):
	title, image = title_and_image
	fig = plt.subplot(subplot_rows,subplot_cols,index+1)
	plt.imshow(image, cmap='gray')
	plt.title(title)
	fig.axes.get_yaxis().set_ticks([])
	fig.axes.get_xaxis().set_ticks([])
plt.show()
