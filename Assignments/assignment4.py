''' Gaussian Filtering for Image with Salt and Pepper Noise
Note: Only for GRAYSCALE IMAGES
'''

import numpy as np
from math import ceil
import sys

# Add salt and pepper noise to image. (Note: percent in range [0 ... 1])
def salt_pepper_noise(image, percent):

	# Add noise to copy of the image
	noisy_image = image.copy()

	if percent == 0:
		return noisy_image

	randint_high_excl = ceil(2/percent)
	random_integers = np.random.randint(0, randint_high_excl, size=image.shape)

	noisy_image[np.where(random_integers==0)] = 0
	noisy_image[np.where(random_integers==(randint_high_excl-1))] = 255

	return noisy_image


# Perform Gaussian filtering of image
def gaussian_filtering(image, sigma=1, filter_size=3):

	# Note: Filter size must be ODD
	if filter_size%2==0:
		print('Error: The filter size must be ODD')
		sys.exit(1)

	# Note: Works only for grayscale images
	if len(image.shape)>2:
		print('Not Implemented:  Gaussian Filtering not implemented for color images')
		sys.exit(1)

	# Pad the image with appropriate zeros on all sides
	zero_padded_img = np.zeros((image.shape[0]+filter_size-1, image.shape[1]+filter_size-1), dtype=image.dtype)
	zero_padded_img[filter_size//2:(zero_padded_img.shape[0]-filter_size//2), filter_size//2:(zero_padded_img.shape[1]-filter_size//2)] = image

	# Generate sizexsize gaussian filter
	guassian_filter = np.empty((filter_size, filter_size))

	for x_index, x in enumerate(range(-1*(filter_size//2), filter_size//2+1)):
		for y_index, y in enumerate(range(-1*(filter_size//2), filter_size//2+1)):
			guassian_filter[x_index, y_index] = np.exp(-1*(x*x + y*y)/(2*sigma*sigma))

	guassian_filter /= (2*np.pi*sigma*sigma)
	
	# transform filter s.t. sum of elements in filter = 1
	guassian_filter /= np.sum(guassian_filter)

	# Filter image using gaussian filter
	filtered_img = np.empty(image.shape, dtype=image.dtype)

	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			filtered_img[x,y] = np.sum(zero_padded_img[x:x+filter_size, y:y+filter_size]*guassian_filter)

	return filtered_img


if __name__ == '__main__':

	print('Instruction: Press \'ENTER\' to close an image and continue ...\n'
		'Note: Don\'t close the image window using \'x\' button\n\n'
		'Press ENTER to continue ...',end='')
	input()

	''' Load the Lena image of size 512 x 512 with 256 levels '''

	# import necessary libraries - OpenCV, numpy, matplotlib
	import cv2 as cv

	# Read the lena image as a 'BGR' color image
	gray_img = cv.imread('lena_std.tif', cv.IMREAD_GRAYSCALE)
	cv.imshow('Lena Image', gray_img)

	# Display image until enter is pressed
	while cv.waitKey(0)!=13:
		pass

	cv.destroyWindow('Lena Image')

	print('Enter percentage of salt and pepper noise (range: [0 ... 1]) : ', end='')
	noise_percent = float(input())

	# Add salt and pepper noise to image
	noisy_gray_img = salt_pepper_noise(gray_img, noise_percent)
	cv.imshow('Noisy Lena Image', noisy_gray_img)
	
	# Display image until enter is pressed
	while cv.waitKey(0)!=13:
		pass
	
	cv.destroyWindow('Noisy Lena Image')

	print('Enter value of sigma for gaussian: ', end='')
	sigma = float(input())

	print('Enter size of filter: ', end='')
	filter_size = int(input())

	# Perform Gaussian Filtering of Noisy image
	filtered_img = gaussian_filtering(noisy_gray_img, sigma=sigma, filter_size=filter_size)
	cv.imshow('Filtered Lena Image', filtered_img)
	
	# Display image until enter is pressed
	while cv.waitKey(0)!=13:
		pass
	
	cv.destroyWindow('Filtered Lena Image')

	cv.imwrite('gaussian_sp_noise/simga={},filter={},noise={}.tif'.format(sigma, filter_size, noise_percent), filtered_img)