import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import *



# load data, grayscale then binary, crop, invert color
img_gray = cv2.imread('data/images/test_4.png', cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bw_crop = crop_borders(im_bw)
img_pre = cv2.bitwise_not(img_bw_crop)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (199, 199))
img = cv2.morphologyEx(img_pre, cv2.MORPH_CLOSE, kernel)

im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create hull array for convex hull points
hull = []
 
# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))

# create an empty black image
drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
color_ch = (0, 255, 0)  # blue - color for convex hull
#     color_contours = (0, 0,0 )  # green - color for contours

# draw contours and hull points
# for i in range(len(contours)):
#     # draw ith contour
#     cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
# draw 0th convex hull object
cv2.drawContours(drawing, hull, 0, color_ch, 1, 8)

#count points belonging to CH
count = 0
for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        pt_location = cv2.pointPolygonTest(hull[0], (row, col), False)
        if pt_location >= 0:
            count += 1


print(
    f'Pixels in CH: {count}\nPixels in Panicle: {len(np.nonzero(img_pre)[1])}\nRatio: {len(np.nonzero(img_pre)[1])/count}')


# visualization
visualize = 1
images = [
        ('Input', img_pre),
        ('Morphology: Closing', img),
		('Convex Hull', drawing),
]
if visualize:
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
