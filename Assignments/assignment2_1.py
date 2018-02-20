import numpy as np
from math import ceil

def scaling_nn(img, height_factor, width_factor=None):
    ''' Image Scaling using Nearest Neighbor interpolation '''

    if width_factor is None:
        width_factor = height_factor

    # shape of scaled 'GRAYSCALE' image
    if len(img.shape)==2:
        scaled_img_shape = (ceil(img.shape[0]*height_factor), ceil(img.shape[1]*width_factor)) 

    # shape of scaled 'BGR' (alpha ignored) image
    else:
        scaled_img_shape = (ceil(img.shape[0]*height_factor), ceil(img.shape[1]*width_factor), 3)

    scaled_img = np.empty(scaled_img_shape, dtype=img.dtype)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            scaled_img[ceil(x*height_factor):ceil((x+1)*height_factor), ceil(y*width_factor):ceil((y+1)*width_factor)] = img[x, y]

    return scaled_img



if __name__ == '__main__':
    
    # importing necessary libraries - OpenCV
    import cv2 as cv

    # Read the lena image as a 'BGR' color image
    img = cv.imread('lena_std.tif', cv.IMREAD_COLOR)

    # height and width scaling factor pairs
    scaling_factors = [(1.5, 1.5), (1.25, 0.75), (0.75, 0.75), (1, 2), (0.25, 0.25), (4, 4)]

    for scaling_factor in scaling_factors:

        # generate scaled image using Nearest Neighbor interpolation
        scaled_img = scaling_nn(img, *scaling_factor)
        cv.imwrite('nn_interpolation/scaled_{}x{}.tif'.format(scaled_img.shape[0], scaled_img.shape[1]), scaled_img)
