from math import log2, log10, sqrt
import cv2
import numpy as np
import math


def PSNR(original, compressed):
	""" This function computes the peak signal to noise ration of two images
	Args:
	original   : original frame before conversion
	compressed : frame after conversion
	Returns:
	psnr     : peak signal to noise ration
	"""
	#original   = original[0:280, 0:280]
	#compressed = compressed[0:280, 0:280]

	mse = np.mean((original - compressed)**2)
	

	if mse == 0:                        # MSE = 0 means no noise is present in the original signal
		return 100

	max_pixel = 255.0

	psnr = 10*log10((max_pixel**2) / mse)

	return psnr