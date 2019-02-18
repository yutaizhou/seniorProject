import numpy as np
import itertools
import matplotlib.pyplot as plt
import cv2
from scipy.sparse.dok import dok_matrix
from scipy.sparse.csgraph import dijkstra

def compute_main_axis_length(img):
	rows, cols = img.shape
	adjacency = make_adjacency_matrix(img)

	source_index = to_index(cols, 2746, 1141)
	target_index = to_index(cols, 6, 878)

	dist_matrix, predecessors = dijkstra(adjacency, directed=False, indices=[source_index],
						   				 unweighted=True, return_predecessors=True)

	pixel_index = target_index
	pixels_path = []
	while pixel_index != source_index:
		pixels_path.append(pixel_index)
		pixel_index = predecessors[0, pixel_index]

	img_path = np.zeros(img.shape)
	for pixel_index in pixels_path:
		row, col = to_coordinates(cols, pixel_index)
		img_path[row, col] = 1

	return np.sum(img_path), img_path


def to_index(cols, row, col):
	return row * cols + col


def to_coordinates(cols, index):
	return index // cols, index % cols


def make_adjacency_matrix(img):
	rows, cols = img.shape
	adjacency = dok_matrix((rows * cols, rows * cols), dtype=bool)

	directions = list(itertools.product([0, 1, -1], [0, 1, -1]))
	for row in range(1, rows - 1):
		for col in range(1, cols - 1):
			if not img[row, col]:
				continue

			for y_diff, x_diff in directions:
				if img[row + y_diff, col + x_diff]:
					adjacency[to_index(cols, row, col), to_index(cols, row + y_diff, col + x_diff)] = True
	return adjacency
