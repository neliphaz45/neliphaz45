from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2

def compare(filename1, filename2):

	img1      = cv2.imread(filename1)
	img2      = cv2.imread(filename2)

	gray1     = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2     = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	s         = ssim(gray1, gray2)

	return s