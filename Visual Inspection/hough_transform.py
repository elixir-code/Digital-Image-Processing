''' Python Implementation of Circular Hough Transform 
	
Reference: 	Extract the Geometry of Mechanical Parts by Vision System Using Hough Transform
			- A. A. Alshennawy
'''

import cv2 as cv
import numpy as np

from math import floor

def gen_frustum_template(rmin, rmax, rstep):

	frustum_template = np.zeros((2*rmax+1, 2*rmax+1, floor((rmax - rmin)/rstep)+1), dtype=np.uint8)
	
	# iterate through all radii and draw circles
	radius = rmin
	layer_index = 0

	circle_template = np.empty((2*rmax+1, 2*rmax+1), dtype=np.uint8)

	while radius <= rmax:
		
		circle_template.fill(0)
		cv.circle(circle_template, (rmax, rmax), radius, 1, 1)
		frustum_template[:,:,layer_index] = circle_template

		radius += rstep
		layer_index += 1

	return frustum_template


def comp_accumulator_values(edge_img, rmax, frustum_template):
	
	accumulator = np.zeros((edge_img.shape[0], edge_img.shape[1], frustum_template.shape[2]), dtype=np.uint32)

	for i in range(edge_img.shape[0]):
		for j in range(edge_img.shape[1]):

			# image[i,j] is an edge point
			if edge_img[i,j] > 127:
				accumulator[max(i-rmax,0):min(i+rmax+1, edge_img.shape[0]),max(j-rmax,0):min(j+rmax+1, edge_img.shape[1])] += frustum_template[max(rmax-i, 0):rmax+min(edge_img.shape[0]-i,rmax+1), max(rmax-j, 0):rmax+min(edge_img.shape[1]-j,rmax+1)]

	return accumulator