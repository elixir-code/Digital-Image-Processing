import numpy as np
from math import floor, ceil

def scaling_bl(img, height_factor, width_factor=None):

    if width_factor is None:
        width_factor = height_factor

    # shape of scaled 'GRAYSCALE' image
    if len(img.shape)==2:
        scaled_img_shape = (ceil(img.shape[0]*height_factor), ceil(img.shape[1]*width_factor)) 

    # shape of scaled 'BGR' (or 'BGR' with aplha) image
    else:
        scaled_img_shape = (ceil(img.shape[0]*height_factor), ceil(img.shape[1]*width_factor), img.shape[2])

    # Note: The size of scaled image should be greater than 1 pixel in x any y dim
    inverse_factor_height = (img.shape[0]-1)/(scaled_img_shape[0]-1)
    inverse_factor_width = (img.shape[1]-1)/(scaled_img_shape[1]-1)

    # Compute the interpolated index values of scaled array
    interpolated_indices = np.empty(scaled_img_shape[:2]+(2,))

    for index in range(scaled_img_shape[0]):
        interpolated_indices[index, :, 0] = index*inverse_factor_height

    for index in range(scaled_img_shape[1]):
        interpolated_indices[:, index, 1] = index*inverse_factor_width

    # Compute bi-linear interpolation value
    def bl_inter_value(index, img=img):
        boundary_pixel1 = img[floor(index[0]), floor(index[1])]
        boundary_pixel2 = img[floor(index[0]), ceil(index[1])]
        boundary_pixel3 = img[ceil(index[0]), floor(index[1])]
        boundary_pixel4 = img[ceil(index[0]), ceil(index[1])]

        index_fraction = index%1

        return np.uint8( (1-index_fraction[0])*(1-index_fraction[1])*boundary_pixel1 + (1-index_fraction[0])*index_fraction[1]*boundary_pixel2 + index_fraction[0]*(1-index_fraction[1])*boundary_pixel3 + index_fraction[0]*index_fraction[1]*boundary_pixel4 )

    bl_interpolation = np.vectorize(bl_inter_value, signature='(2)->({})'.format(1 if len(img.shape)==2 else img.shape[2]))
    scaled_img = bl_interpolation(interpolated_indices)

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
        scaled_img = scaling_bl(img, *scaling_factor)
        cv.imwrite('bl_interpolation/scaled_{}x{}.tif'.format(scaled_img.shape[0], scaled_img.shape[1]), scaled_img)