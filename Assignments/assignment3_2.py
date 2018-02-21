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


# Perform histogram matching of a 'image' with reference to 'ref_image'
def histogram_matching(image, ref_image):

	# compute histogram equalisation transformation array for image and reference image
	image_hist_equ_trans = comp_hist_equ_trans(image)
	ref_image_hist_equ_trans = comp_hist_equ_trans(ref_image)

	# The histogram matching transformation of an image with reference to another image
	image_hist_match_trans = np.empty(256, dtype=np.uint8)

	for pixel_value, trans_pixel_value in enumerate(image_hist_equ_trans):
		image_hist_match_trans[pixel_value] = np.argmin(np.abs(ref_image_hist_equ_trans - trans_pixel_value))

	transformed_image = np.empty(image.shape, dtype=image.dtype)

	# Transform the image
	for x in range(image.shape[0]):
		for y in range(image.shape[1]):
			transformed_image[x, y] = image_hist_match_trans[image[x, y]]

	return transformed_image


import matplotlib.pyplot as plt

# Plot the histograms of original image, reference image and transformed images
def plot_histograms(image, ref_image, matched_image, title='Image'):

	plt.subplot(311)
	plt.title(title)

	plt.xlabel('pixel value (r)')
	plt.ylabel('pf(r)')
	plt.bar(range(256), compute_norm_histogram(image), width=0.8)


	plt.subplot(312)
	plt.title('Reference Image')

	plt.xlabel('pixel value (r)')
	plt.ylabel('pf(r)')
	plt.bar(range(256), compute_norm_histogram(ref_image), width=0.8)


	plt.subplot(313)
	plt.title('Matched {}'.format(title))

	plt.xlabel('pixel value (r)')
	plt.ylabel('pf(r)')
	plt.bar(range(256), compute_norm_histogram(matched_image), width=0.8)

	plt.show()


if __name__ == '__main__':

	# import necessary libraries - OpenCV, Matplotlib
	import cv2 as cv

	''' Load the various pollen images - dark, light and low-contrast 
	Figure 3.16 (Digital Image Processing - Gonzalez)
	'''
	dark_img = cv.imread('pollen_dark.tif', cv.IMREAD_GRAYSCALE)
	light_img = cv.imread('pollen_light.tif', cv.IMREAD_GRAYSCALE)
	low_contrast_img = cv.imread('pollen_low_contrast.tif', cv.IMREAD_GRAYSCALE)
	
	reference_img = cv.imread('pollen_high_contrast.tif', cv.IMREAD_GRAYSCALE)

	
	# Display DARK pollen image
	cv.imshow('Dark Image', dark_img)
	cv.waitKey(0)
	cv.destroyWindow('Dark Image')

	# Perform histogram matching of DARK image with reference image
	hist_matched_image = histogram_matching(dark_img, reference_img)
	cv.imwrite('histogram/matching/dark_matched.tif', hist_matched_image)

	cv.imshow('Matched Image (Dark Image)', hist_matched_image)
	cv.waitKey(0)
	cv.destroyWindow('Matched Image (Dark Image)')

	# Display Histogram of DARK, reference and MATCHED DARK images
	plot_histograms(dark_img, reference_img, hist_matched_image, title='Dark Image')


	# Display LIGHT pollen image
	cv.imshow('Light Image', light_img)
	cv.waitKey(0)
	cv.destroyWindow('Light Image')

	# Perform histogram matching of LIGHT image with reference image
	hist_matched_image = histogram_matching(light_img, reference_img)
	cv.imwrite('histogram/matching/light_matched.tif', hist_matched_image)

	cv.imshow('Matched Image (Light Image)', hist_matched_image)
	cv.waitKey(0)
	cv.destroyWindow('Matched Image (Light Image)')

	# Display Histogram of LIGHT, reference and MATCHED DARK images
	plot_histograms(light_img, reference_img, hist_matched_image, title='Light Image')


	# Display LOW CONTRAST pollen image
	cv.imshow('Low Contrast Image', low_contrast_img)
	cv.waitKey(0)
	cv.destroyWindow('Low Contrast Image')

	# Perform histogram matching of LIGHT image with reference image
	hist_matched_image = histogram_matching(low_contrast_img, reference_img)
	cv.imwrite('histogram/matching/low_contrast_matched.tif', hist_matched_image)

	cv.imshow('Matched Image (Low Contrast Image)', hist_matched_image)
	cv.waitKey(0)
	cv.destroyWindow('Matched Image (Low Contrast Image)')

	# Display Histogram of LOW CONTRAST, reference and MATCHED DARK images
	plot_histograms(low_contrast_img, reference_img, hist_matched_image, title='Low Contrast Image')