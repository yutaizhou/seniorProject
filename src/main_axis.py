import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import *
from skimage.morphology import thin


if len(sys.argv) < 3:
	sys.exit('ERROR! Please execute the program with the following format: python main_axis.py [path_to_image] [erosion_kernel_size]')
img_path = sys.argv[1]

# load data, grayscale then binary, crop, invert color
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
(thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img_bw_crop = crop_borders(img_bw)
img = cv2.bitwise_not(img_bw_crop)

# opening to remove non-main axis
erosion_size = int(sys.argv[2]) 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# connected component to find main axis, then find component containing main axis
_, labels = cv2.connectedComponents(img_open)
img_ccomp = visualize_connected_components(labels)

_, counts = np.unique(labels, return_counts=True)
second_most_common_label = np.argwhere(counts == np.sort(counts)[-2]).squeeze()
binary_main_axis = labels == second_most_common_label
img_main_axis = cv2.bitwise_and(img, img, mask = binary_main_axis.astype('uint8'))


# calculate main axis diameter
diameter_pixels = np.mean(binary_main_axis[-1:-40:-1,:]) * np.float64(binary_main_axis.shape[1])


# calculate main axis length
# first thin
img_main_axis_thin = thin(binary_main_axis).astype('uint8')

# then use dijkstra
length_pixels, img_length = compute_main_axis_length(img_main_axis_thin)

print(f'Diameter: {diameter_pixels}\nLength: {length_pixels}\n')

# visualization
visualize = 1
images = [
		#   ('Input Grayscale ', img_gray),
		#   ('Input Binary', img_bw),
		#   ('Binary Cropped', img_bw_crop),
		  ('Preprocessed', img),
		#   ('Morphology: Opening', img_open),
		#   ('Connected Components', img_ccomp),
		  ('Main Axis', img_main_axis),
		#   ('Main Axis Skeleton', img_main_axis_thin),
		  ('Shortest Path', img_length)
		  ]
if visualize:
	subplot_rows = 1
	subplot_cols = len(images)
	for index, title_and_image in enumerate(images):
		title, image = title_and_image
		fig = plt.subplot(subplot_rows,subplot_cols,index+1)
		plt.imshow(image, cmap='gray')
		plt.title(title)
		fig.axes.get_yaxis().set_ticks([])
		fig.axes.get_xaxis().set_ticks([])
	fig = plt.gcf()
	fig.set_size_inches(18.5, 10.5)
	plt.show()
