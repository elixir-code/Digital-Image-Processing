import numpy as np

# Compute normalised histogram of image
def compute_norm_histogram(image):

	histogram = np.zeros(256, dtype=np.int)

	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			histogram[ image[x,y] ] += 1

	normalised_hist = histogram/(image.shape[0]*image.shape[1])

	return normalised_hist


# Compute the histogram equalisation transformation array from 'grayscale' image
def comp_hist_equ_trans(image):

	# Compute normalised histogram of the image
	normalised_hist = compute_norm_histogram(image)

	# compute cumulative histogram of the image
	cumulative_hist = normalised_hist.copy()
	
	for i in range(1, 256):
		cumulative_hist[i] = cumulative_hist[i] + cumulative_hist[i-1]

	# Generate the pixel intensity transformation
	transformed_intensity = (cumulative_hist * 255).astype(np.uint8)

	return transformed_intensity


# Perform inplace HISTORAM EQUALISATION of the 'grayscale' image
def equalise_histogram(image):

	# Generate the pixel intensity transformation
	transformed_intensity = comp_hist_equ_trans(image)

	# Transform the image
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			image[x, y] = transformed_intensity[image[x, y]]

	return image

import matplotlib.pyplot as plt

# Display image and associated histogram
def display_img_hist(image, title='Image'):

	
	plt.subplot(121)
	
	plt.title(title)
	plt.axis('off')
	plt.imshow(image, cmap='gray')


	plt.subplot(122)
	
	plt.title('Histogram ({})'.format(title))
	plt.xlabel('pixel intensity')
	plt.ylabel('probability(pixel internsity)')
	plt.bar(range(256), compute_norm_histogram(image), width=0.8)

	plt.show()


if __name__ == '__main__':

	# import neccessary libraries - openCV
	import cv2 as cv

	''' Load the various pollen images - dark, light and low-contrast 
	Figure 3.16 (Digital Image Processing - Gonzalez)
	'''
	dark_img = cv.imread('pollen_dark.tif', cv.IMREAD_GRAYSCALE)
	light_img = cv.imread('pollen_light.tif', cv.IMREAD_GRAYSCALE)
	low_contrast_img = cv.imread('pollen_low_contrast.tif', cv.IMREAD_GRAYSCALE)
	high_contrast_img = cv.imread('pollen_high_contrast.tif', cv.IMREAD_GRAYSCALE)

	''' Histogram Equalisation of DARK, LIGHT and LOW CONTRAST IMAGES '''

	# Display original DARK IMAGE and associated histogram
	display_img_hist(dark_img, title='Dark Image')

	# Perform Histogram equalisation of DARK image and display transformed image
	transformed_img = equalise_histogram(dark_img)
	cv.imwrite('histogram/equalisation/dark_equalised.tif', transformed_img)
	display_img_hist(transformed_img, title='Equalised Dark Image')



	# Display original LIGHT IMAGE and associated histogram
	display_img_hist(light_img, title='Light Image')

	# Perform Histogram equalisation of LIGHT image and display transformed image
	transformed_img = equalise_histogram(light_img)
	cv.imwrite('histogram/equalisation/light_equalised.tif', transformed_img)
	display_img_hist(transformed_img, title='Equalised Light Image')



	# Display original LOW CONTRAST IMAGE and associated histogram
	display_img_hist(low_contrast_img, title='Low Contrast Image')

	# Perform Histogram equalisation of LOW CONTRAST image and display transformed image
	transformed_img = equalise_histogram(low_contrast_img)
	cv.imwrite('histogram/equalisation/low_contrast_equalised.tif', transformed_img)
	display_img_hist(transformed_img, title='Equalised Low Contrast Image')



	# Display original HIGH CONTRAST IMAGE and associated histogram
	display_img_hist(high_contrast_img, title='High Contrast Image')

	# Perform Histogram equalisation of LOW CONTRAST image and display transformed image
	transformed_img = equalise_histogram(high_contrast_img)
	cv.imwrite('histogram/equalisation/high_contrast_equalised.tif', transformed_img)
	display_img_hist(transformed_img, title='Equalised High Contrast Image')