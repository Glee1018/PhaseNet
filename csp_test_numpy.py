from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from steerable.SCFpyr_NumPy import SCFpyr_NumPy
import steerable.utils as utils
import cv2
import torch
import torchvision
# img_size=256
img=cv2.imread("./assets/lena.jpg",0)
# img=cv2.resize(img,(img_size,img_size))
# cv2.imshow('img',img)

# build csp
height=12
nbands=4
scale_factor=2**(1/2)
# scale_factor=2
pyr=SCFpyr_NumPy(height=height,nbands=nbands,scale_factor=scale_factor)
# coeff=pyr.build_c(img)
# img_con=pyr.reconstruct_c(coeff)
coeff=pyr.build(img)
img_con=pyr.reconstruct(coeff)
SCFpyr_NumPy.pyr_info(coeff)

print('MSE: ',np.mean(np.power(img.astype(np.float)-img_con,2)))

# coeff_grid = utils.make_grid_coeff(coeff, normalize=True)
# cv2.imshow('coeff', coeff_grid)
# cv2.imshow('img_recon',img_con.astype(np.uint8))
# cv2.waitKey(0)