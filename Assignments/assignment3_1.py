''' Load the pollen images - dark, light and low-contrast '''

import cv2 as cv
import numpy as np

# Assumption -- No. of levels = 256 (0 ... 255)
# Perform HISTORAM EQUALISATION of the image
def equalise_histogram(image):

	# Compute normalised histogram of the image
	histogram = np.zeros(256, dtype=np.int)

	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			histogram[ image[x,y] ] += 1

	normalised_hist = histogram/(image.shape[0]*image.shape[1])

	# compute cumulative histogram of the image
	cumulative_hist = normalised_hist.copy()
	
	for i in range(1, 256):
		cumulative_hist[i] = cumulative_hist[i] + cumulative_hist[i-1]

	# Generate the pixel intensity transformation
	transformed_intensity = cumulative_hist * 255

	# Transform the image
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			image[x, y] = transformed_intensity[image[x, y]]

	return image


dark_img = cv.imread('pollen_dark.tif', cv.IMREAD_GRAYSCALE)
light_img = cv.imread('pollen_light.tif', cv.IMREAD_GRAYSCALE)
low_contrast_img = cv.imread('pollen_low_contrast.tif', cv.IMREAD_GRAYSCALE)
high_contrast_img = cv.imread('pollen_high_contrast.tif', cv.IMREAD_GRAYSCALE)

''' Histogram Equalisation of DARK, LIGHT and LOW CONTRAST IMAGES '''

# Display original DARK IMAGE
cv.imshow('DARK IMAGE', dark_img)
cv.waitKey(0)
cv.destroyWindow('DARK IMAGE')

# Perform Histogram equalisation of DARK image and display transformed image
transformed_img = equalise_histogram(dark_img)

cv.imshow('HISTOGRAM EQUALISED IMAGE', transformed_img)
cv.waitKey(0)
cv.destroyWindow('HISTOGRAM EQUALISED IMAGE')



# Display original LIGHT IMAGE
cv.imshow('LIGHT IMAGE', light_img)
cv.waitKey(0)
cv.destroyWindow('LIGHT IMAGE')

# Perform Histogram equalisation of LIGHT image and display transformed image
transformed_img = equalise_histogram(light_img)

cv.imshow('HISTOGRAM EQUALISED IMAGE', transformed_img)
cv.waitKey(0)
cv.destroyWindow('HISTOGRAM EQUALISED IMAGE')



# Display original LOW CONTRAST IMAGE
cv.imshow('LOW CONTRAST IMAGE', low_contrast_img)
cv.waitKey(0)
cv.destroyWindow('LOW CONTRAST IMAGE')

# Perform Histogram equalisation of LOW CONTRAST image and display transformed image
transformed_img = equalise_histogram(low_contrast_img)

cv.imshow('HISTOGRAM EQUALISED IMAGE', transformed_img)
cv.waitKey(0)
cv.destroyWindow('HISTOGRAM EQUALISED IMAGE')


# Display original HIGH CONTRAST IMAGE
cv.imshow('HIGH CONTRAST IMAGE', high_contrast_img)
cv.waitKey(0)
cv.destroyWindow('HIGH CONTRAST IMAGE')

# Perform Histogram equalisation of HIGH CONTRAST image and display transformed image
transformed_img = equalise_histogram(high_contrast_img)

cv.imshow('HISTOGRAM EQUALISED IMAGE', transformed_img)
cv.waitKey(0)
cv.destroyWindow('HISTOGRAM EQUALISED IMAGE')