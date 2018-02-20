from math import ceil, floor
import numpy as np

# Compute dimensions of theta (radians) rotated image
def comp_rotated_shape(img_shape, rotation_angle):

	if rotation_angle>=0 and rotation_angle<=np.pi/2:
		rotated_img_width = ceil(img_shape[0]*np.sin(rotation_angle) + img_shape[1]*np.cos(rotation_angle)) + 1
		rotated_img_height = ceil(img_shape[0]*np.cos(rotation_angle) + img_shape[1]*np.sin(rotation_angle)) + 1

	elif rotation_angle>np.pi/2 and rotation_angle<=np.pi:
		rotated_img_width = ceil(img_shape[0]*np.sin(rotation_angle) - img_shape[1]*np.cos(rotation_angle)) + 1
		rotated_img_height = ceil(img_shape[1]*np.sin(rotation_angle) - img_shape[0]*np.cos(rotation_angle)) + 1

	elif rotation_angle>np.pi and rotation_angle <= 3*np.pi/2:
		rotated_img_width = ceil(-1*(img_shape[0]*np.sin(rotation_angle) + img_shape[1]*np.cos(rotation_angle))) + 1
		rotated_img_height = ceil(-1*(img_shape[0]*np.cos(rotation_angle) + img_shape[1]*np.sin(rotation_angle))) + 1

	else:
		rotated_img_width = ceil(img_shape[1]*np.cos(rotation_angle) - img_shape[0]*np.sin(rotation_angle)) + 1
		rotated_img_height = ceil(img_shape[0]*np.cos(rotation_angle) - img_shape[1]*np.sin(rotation_angle)) + 1

	return rotated_img_height, rotated_img_width

# Compute bi-linear interpolation pixel values from pixel indices
def bl_inter_value(index, img):

	# Fill with black pixels for indices outside image
	if (index[0] < 0 or (index[0] > img.shape[0]-1)) or (index[1] < 0 or index[1] > (img.shape[1]-1)):
		return np.array([0]*(1 if len(img.shape)==2 else img.shape[2]), dtype=np.uint8)

	else:
		boundary_pixel1 = img[floor(index[0]), floor(index[1])]
		boundary_pixel2 = img[floor(index[0]), ceil(index[1])]
		boundary_pixel3 = img[ceil(index[0]), floor(index[1])]
		boundary_pixel4 = img[ceil(index[0]), ceil(index[1])]

		index_fraction = index%1

		return np.uint8( (1-index_fraction[0])*(1-index_fraction[1])*boundary_pixel1 + (1-index_fraction[0])*index_fraction[1]*boundary_pixel2 + index_fraction[0]*(1-index_fraction[1])*boundary_pixel3 + index_fraction[0]*index_fraction[1]*boundary_pixel4 )

# Rotate the given image by theta degrees (in anti-clockwise direction)
def rotation_bl(img, theta_degree):

	# convert rotation angle to radians
	rotation_angle = theta_degree*np.pi/180

	# rotation has periodicity of 2*pi
	rotation_angle = rotation_angle%(2*np.pi)

	if rotation_angle < 0:
		rotation_angle = 2*pi - rotation_angle

	print("Rotation Angle = {} degrees ({:.2f}*pi radians)".format(theta_degree, rotation_angle/np.pi))

	rotated_img_height, rotated_img_width = comp_rotated_shape(img.shape[:2], rotation_angle)

	# Estimate the dimension of the rotated image : width and height
	print("Rotated Image Dim = ({},{})\n".format(rotated_img_height, rotated_img_width))

	# The interpolation indices after rotation
	rot_center_x, rot_center_y = rotated_img_height//2, rotated_img_width//2
	
	# calculate center pixel of original image
	orig_center_x, orig_center_y = img.shape[0]//2, img.shape[1]//2

	rot_inter_indices = np.empty((rotated_img_height, rotated_img_width, 2), dtype=np.int)

	for i, pixel_index in enumerate( range(rot_center_x, rot_center_x - rotated_img_height, -1) ):
		rot_inter_indices[i,:,0] = pixel_index

	for j, pixel_index in enumerate( range(-1*rot_center_y, rotated_img_width - rot_center_y) ):
		rot_inter_indices[:,j,1] = pixel_index

	# Compute 'rotated and interpolated' pixel values
	def bl_rot_inter_value(pixel_index, img=img, rotation_angle=rotation_angle, orig_center_x=orig_center_x, orig_center_y=orig_center_y):
	
		interpolated_index = np.array( 
										[
											orig_center_x - (pixel_index[0]*np.cos(rotation_angle) - pixel_index[1]*np.sin(rotation_angle)), 
											pixel_index[0]*np.sin(rotation_angle) + pixel_index[1]*np.cos(rotation_angle) + orig_center_y
										]
									)

		return bl_inter_value(interpolated_index, img)

	bl_rot_interpolation = np.vectorize(bl_rot_inter_value, signature='(2)->({})'.format(1 if len(img.shape)==2 else img.shape[2]))

	# bl_rot_interpolation = np.vectorize(bl_rot_inter_value, signature='(2)->(2)')
	rotated_img = bl_rot_interpolation(rot_inter_indices)

	return rotated_img


if __name__ == '__main__':
	
	# import necessary libraries - OpenCV
	import cv2 as cv

	# Read the LEANING TOWER OF PISA image as a 'BGR' color image
	img = cv.imread('pisa_tower.jpeg', cv.IMREAD_COLOR)
	print("Original Image Shape = ",img.shape, end='\n\n')


	''' Rotation and Interpolation - Nearest Neighbor Interpolation '''

	# Rotate image from 0 degree to 10 degree in steps of 1 degrees (anti-clockwise rotation)
	for rotation_angle_deg in range(1, 11, 1):
		rotated_img = rotation_bl(img, rotation_angle_deg)
		cv.imwrite('bl_rotation/rotated_{}degrees.jpeg'.format(rotation_angle_deg), rotated_img)