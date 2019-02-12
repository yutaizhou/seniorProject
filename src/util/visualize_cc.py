import cv2
import numpy as np

def visualize_connected_components(img_labeled):
	label_hue = np.uint8(179*img_labeled/np.max(img_labeled))
	blank_ch = 255*np.ones_like(label_hue)
	img_ccomp = cv2.merge([label_hue, blank_ch, blank_ch])
	img_ccomp = cv2.cvtColor(img_ccomp, cv2.COLOR_BGR2HSV)
	img_ccomp[label_hue==0] = 0

	return img_ccomp
