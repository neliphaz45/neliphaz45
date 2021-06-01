Attention Mechanism (as implemented in Vision Transformer) for Image compression


The encoder takes the original image (in this case we used 3x224x224 images), divides it into fixed size patches, adds positional information to each patch, applies dropout and then passes the image through a cascade of 6 blocks of a Transformer module followed by layer normalization and the linear mapping.
We use a patch size of 16 x 16 and each patch is flattened into a vector of size 1x768 (3x16x16). 
As a result, each 224x224 image is divided into 196 patches and reshaped into a 196x768 matrix. The embedding size (768) and the
number of patches are maintained throughout the encoder module until the final linear layer. This linear layer is the one
that does the compression, converting each image’s representation to a 1x1176 vector. Thus, the encoder compresses the
image from 3x224x224x8 bits to 1176x64 bits which is 0.5 bits per pixel. 

For the decoder, we use one linear layer to expand the compressed representation from a vector of size
1176 to a vector of size 150528 which is then reshaped into a tensor of shape (3x224x224).
