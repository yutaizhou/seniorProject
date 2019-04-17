import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import *


if len(sys.argv) < 2:
	sys.exit('ERROR! Please execute the program with the following format: python density.py [path_to_image]')
img_path = sys.argv[1]

# load data, grayscale then binary, crop, invert color
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bw_crop = crop_borders(im_bw)
img = cv2.bitwise_not(img_bw_crop)

# closing to fill in gaps of panicle
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (199, 199))
img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# find contours
contours, _ = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# create convex hull for each contour
convex_hulls = []
for i in range(len(contours)):
    convex_hulls.append(cv2.convexHull(contours[i], False))

# draw convex hull object
num_hulls = len(convex_hulls)
subplot_rows = 3
subplot_cols = np.ceil((num_hulls + 1) / subplot_rows)

fig = plt.subplot(subplot_rows, subplot_cols, 1)
plt.imshow(img_close, cmap='gray')
plt.title('Preprocessed Input')
fig.axes.get_yaxis().set_ticks([])
fig.axes.get_xaxis().set_ticks([])

for index, convex_hull in enumerate(convex_hulls):
	fig = plt.subplot(subplot_rows, subplot_cols, index+2)
	drawing = np.zeros((img_close.shape[0], img_close.shape[1], 3), np.uint8)
	cv2.drawContours(drawing, convex_hulls, index, (0, 255, 0), 1, 8)
	plt.imshow(drawing, cmap='gray')
	plt.title(index)
	fig.axes.get_yaxis().set_ticks([])
	fig.axes.get_xaxis().set_ticks([])
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()

# count points belonging to CH
hull_index = int(input('Please enter index of the desired convex hull: '))
count = 0
for row in range(img_close.shape[0]):
    for col in range(img_close.shape[1]):
        pt_location = cv2.pointPolygonTest(convex_hulls[hull_index], (row, col), False)
        if pt_location >= 0:
            count += 1


print(f'Pixels in CH: {count}\nPixels in Panicle: {len(np.nonzero(img)[1])}\nRatio: {len(np.nonzero(img)[1])/count * 100:.4}%')


# visualization
visualize = 1

if visualize:
	drawing = np.zeros((img_close.shape[0], img_close.shape[1], 3), np.uint8)
	cv2.drawContours(drawing, convex_hulls, hull_index, (0, 255, 0), 1, 8)

	images = [
        ('Input', img),
        ('Morphology: Closing', img_close),
		('Convex Hull', drawing),
	]

	subplot_rows = 1
	subplot_cols = len(images)
	for index, title_and_image in enumerate(images):
		title, image = title_and_image
		fig = plt.subplot(subplot_rows, subplot_cols, index+1)
		plt.imshow(image, cmap='gray')
		plt.title(title)
		fig.axes.get_yaxis().set_ticks([])
		fig.axes.get_xaxis().set_ticks([])
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	plt.show()
