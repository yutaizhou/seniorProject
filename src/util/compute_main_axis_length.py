import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_main_axis_length(img, img_binary):
	# start_r = np.max(np.argwhere(img_binary),axis = 0)[0]
	# start_c = np.argwhere(img_binary[start_r,:]).squeeze()[0]
	# axis_pixels = [(start_r, start_c)]
	# print(img.shape)
