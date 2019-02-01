import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import *


# load data, grayscale then binary, crop, invert color
img_gray = cv2.imread('data/test_2.png', cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bw_crop = crop_borders(im_bw)
img = cv2.bitwise_not(img_bw_crop)

im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # create hull array for convex hull points
# hull = []
#
# for i in range(len(contours)):
#     hull.append(cv2.convexHull(contours[i], False))
#
# # create an empty black image
# drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
#
# # draw contours and hull points
# for i in range(len(contours)):
#     color_contours = (0, 255, 0) # green - color for contours
#     color = (255, 0, 0) # blue - color for convex hull
#     # draw ith contour
#     # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
#     # draw ith convex hull object
#     cv2.drawContours(drawing, hull, i, color, 1, 8)

hull = cv2.convexHull(img, False)
drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

cv2.drawContours(im2, hull, 0, (255, 0, 0), 1, 8)
plt.imshow(im2)
plt.show()
