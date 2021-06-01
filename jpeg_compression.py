#import numpy as np
from simplejpeg import encode_jpeg, decode_jpeg

from PIL import Image

#frame_n     = np.array(Image.open('bait1.png'))

#print(frame_n.shape)

#compressed  = encode_jpeg(frame_n, 85, 'RGB','444',True)

#decoded     = decode_jpeg(compressed,'RGB',False,False,0,0,1,None,)

#print(decoded.shape)

from os.path import exists
import os
import time
import numpy as np
import torch
import torchvision
from torch import nn , optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image

import random
from torch.utils.data import random_split

random.seed(0)

img_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.24703223,  0.24348513 , 0.26158784))
])	


testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=img_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False, num_workers=2)


for data in test_loader:

	break


import torch
from torchvision import utils as vutils
 
 
def save_image_tensor(input_tensor, filename):

	#print(len(input_tensor.shape))
	input_tensor = input_tensor.clone().detach()

	#print(input_tensor)
	input_tensor = input_tensor.to(torch.device('cpu'))
	#input_tensor = unnormalize(input_tensor)

	vutils.save_image(input_tensor, filename)

for i in range(100):

	save_image_tensor(data[0][i], 'jpeg_50_'+str(i+1)+'.png')

average_time        = 0.0

for j in range(15):

	decode_time     = 0.0

	for i in range(100):

		frame_n     = np.array(Image.open('jpeg_50_'+str(i+1)+'.png'))

		startTime   = time.time()

		compressed  = encode_jpeg(frame_n, 50, 'RGB','444',True)

		endTime     = time.time()
		#startTime   = time.time()

		decoded     = decode_jpeg(compressed,'RGB',False,False,0,0,1,None)

		#endTime     = time.time()

		decode_time = decode_time + (endTime - startTime)

		#L_n                    = L_n.astype(np.uint8)               
		decoded     = Image.fromarray(decoded)
		decoded.save('test_recon_50_'+str(i+1)+'.png')

	average_time = average_time + (decode_time/100)

		#print("Decode time: {}".format(endTime - startTime))
print("Decode time: {}".format(average_time/15))


#for i in range(16):
	
#	save_image_tensor(data[0][i], 'test'+str(i+1)+'.png')

#for i in range(16):

#	startTime = time.time()

#	frame_n     = np.array(Image.open('jpeg_50_'+str(i+1)+'.png'))

#	compressed  = encode_jpeg(frame_n, 50, 'RGB','444',True)

#	decoded     = decode_jpeg(compressed,'RGB',False,False,0,0,1,None)

	#L_n                    = L_n.astype(np.uint8)               
#	decoded     = Image.fromarray(decoded)
#	decoded.save('test_recon'+str(i+1)+'.png')

#end_time = time.time()


#print("Epoch: