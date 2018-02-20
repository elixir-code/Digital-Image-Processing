''' Load the Lena image of size 512 x 512 with 256 levels '''

# import necessary libraries - OpenCV, numpy, matplotlib
import cv2 as cv
import numpy as np
# from matplotlib import pyplot as plt

# Read the lena image as a 'BGR' color image and grayscale image
color_img = cv.imread('lena_std.tif', cv.IMREAD_COLOR)
gray_img = np.mean(color_img, axis=2).astype(np.uint8)

''' Effect of Sampling on Image '''

# Sample image 'img' with factor 2, 4, 8, 16, 32, 64
sampling_factors = [2, 4, 6, 8, 16, 32, 64]

for sampling_factor in sampling_factors:
    
    color_sampled_image = color_img[::sampling_factor,::sampling_factor]

    # write the sampled images
    cv.imwrite('sampling/sampled{}.tif'.format(sampling_factor), color_sampled_image)


''' Effect of Quantisation on Image '''
# Quantize image to 128, 64, 32, 16, 8, 4, 2, 1 levels
# Doubt: How to reconstruct image to display from quantised image

quantisation_levels = [128, 64, 32, 16, 8, 4, 2, 1]
quantisation_factors = [256//quantisation_level for quantisation_level in quantisation_levels]

for quantisation_factor in quantisation_factors:
    
    color_quantised_image = color_img//quantisation_factor
    gray_quantised_image = gray_img//quantisation_factor

    # Use last pixel value of the range as pixel value
    color_reconstructed_image = color_quantised_image*quantisation_factor + quantisation_factor//2
    gray_reconstructed_image = gray_quantised_image*quantisation_factor + quantisation_factor//2
    
    # write the quantised image
    cv.imwrite('quantisation/color/quantised_{}levels.tif'.format(256//quantisation_factor), color_reconstructed_image)
    cv.imwrite('quantisation/grayscale/quantised_{}levels.tif'.format(256//quantisation_factor), gray_reconstructed_image)