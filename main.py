import psnr
import numpy as np
import similarity
from PIL import Image

#original_image_names = ['test'+str(i+1)+'.png' for i in range(16)]
#decoded_image_names  = ['test_recon'+str(i+1)+'.png' for i in range(16)]

original_image_names = ['jpeg_50_'+str(i+1)+'.png' for i in range(100)]
decoded_image_names  = ['test_recon_50_'+str(i+1)+'.png' for i in range(100)]


simm                 = 0.0
ratio                = 0.0

for i in range(len(original_image_names)):

	img1             = original_image_names[i]
	img2             = decoded_image_names[i]

	original         = np.array(Image.open(img1))                  
	decoded          = np.array(Image.open(img2)) 

	noise_ratio      = psnr.PSNR(original, decoded)

	sim              = similarity.compare(original_image_names[i], decoded_image_names[i])

	simm             = simm + sim
	ratio            = ratio + noise_ratio 
	#print(original_image_names[i], "\t", decoded_image_names[i], "\t", "PSNR: ", noise_ratio, "\t", "SSIM: ", sim)

print("Average SSIM: ", simm / 100)
print("Average PSNR: ", ratio / 100)