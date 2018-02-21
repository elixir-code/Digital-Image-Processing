''' Load the Lena image of size 512 x 512 with 256 levels '''

# import necessary libraries - OpenCV, numpy, matplotlib
import cv2 as cv
import numpy as np
# from matplotlib import pyplot as plt

# Read the lena image as a 'BGR' color image and grayscale image
color_img = cv.imread('lena_std.tif', cv.IMREAD_COLOR)


# Add 'Speckle' or 'Salt and Pepper' Noise to image

# percentage of salt and pepper noise to be added
percent_noise = 0.1
random_values = np.random.randint(0, ceil(percent_noise*50))
