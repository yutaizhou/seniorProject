import numpy as np
def crop_borders(img):
	height = img.shape[0]
	borders = np.argwhere((np.sum(img, axis=0) <= (253 * height))).squeeze()
	left_border = borders[0] - 100
	right_border = borders[-1] + 100
	img_cropped = img[:,left_border:right_border]
	return img_cropped
