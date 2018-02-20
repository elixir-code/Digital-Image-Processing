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


# Rotate the given image by theta degrees (in anti-clockwise direction)
def rotation_nn(img, theta_degree):

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

	# Initialise and calculate center pixel of rotated image
	if len(img.shape)==2:
		rotated_img = np.zeros((rotated_img_height, rotated_img_width), dtype=np.uint8)

	else:
		rotated_img = np.zeros((rotated_img_height, rotated_img_width, img.shape[2]), dtype=np.uint8)

	rot_center_x, rot_center_y = rotated_img_height//2, rotated_img_width//2

	# calculate center pixel of original image
	orig_center_x, orig_center_y = img.shape[0]//2, img.shape[1]//2

	# Populate the pixel of rotated image using nearest neighbor interpolation
	for x, center_transformed_x in enumerate( range(-1*rot_center_x, ceil(rotated_img_height/2)) ):
		for y, center_transformed_y in enumerate( range(-1*rot_center_y, ceil(rotated_img_width/2)) ):

			centered_inverse_x = floor(center_transformed_x*np.cos(rotation_angle) + center_transformed_y*np.sin(rotation_angle))
			centered_inverse_y = floor(center_transformed_y*np.cos(rotation_angle) - center_transformed_x*np.sin(rotation_angle))

			inverse_x, inverse_y = centered_inverse_x + orig_center_x, centered_inverse_y + orig_center_y

			if (inverse_x>=0 and inverse_x<img.shape[0]) and (inverse_y>=0 and inverse_y<img.shape[1]):
				rotated_img[x, y] = img[inverse_x, inverse_y]

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
		rotated_img = rotation_nn(img, rotation_angle_deg)
		cv.imwrite('nn_rotation/rotated_{}degrees.jpeg'.format(rotation_angle_deg), rotated_img)


