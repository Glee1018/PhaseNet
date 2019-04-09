from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from steerable.SCFpyr_NumPy import SCFpyr_NumPy
import steerable.utils as utils
import cv2
import torch
import torchvision
img_size=256
img=cv2.imread("./assets/lena.jpg",0)
img=cv2.resize(img,(img_size,img_size))
# cv2.imshow('img',img)

# crop = torchvision.transforms.Resize(512)
# from PIL import Image
# PIL_img=Image.fromarray(img)
# croped_img=crop(PIL_img)
# print(type(croped_img))
# print(croped_img.size)
# croped_img.show()

# build csp
height=12
nbands=4
scale_factor=2**(1/2)
# scale_factor=2
pyr=SCFpyr_NumPy(height=height,nbands=nbands,scale_factor=scale_factor)
coeff=pyr.build_c(img)
# coeff=pyr.build(img)

SCFpyr_NumPy.pyr_info(coeff)
# test pytorch
# coeff_t_r=torch.from_numpy(coeff[0].real)
# coeff_t_i=torch.from_numpy(coeff[0].imag)
# coeff_2=np.empty(shape=(coeff_t_r.numpy()).shape,dtype=np.complex)
# coeff_2.real=coeff_t_r.numpy()
# coeff_2.imag=coeff_t_i.numpy()
# img2=((np.fft.ifft2(np.fft.ifftshift(coeff_2))).real)
# img22=((np.fft.ifft2(np.fft.ifftshift(coeff[0]))).real)
# print(np.mean(np.power(img2-img22,2)))
# # cv2.imshow(' ',(255 * (img2-img2.min())/(img2.max()-img2.min())).astype(np.uint8))
# # cv2.waitKey()


img_con=pyr.reconstruct_c(coeff)
# img_con=pyr.reconstruct(coeff)
print(img_con.max(),img_con.min())
# img_temp=255 * (img_con-img_con.min())/(img_con.max()-img_con.min())
# cv2.imshow('img_recon',img_temp.astype(np.uint8))
# print('MSE(img and img_recon): ',np.mean(np.power(img-img_temp.astype(np.uint8),2)))

cv2.imshow('img_recon',img_con.astype(np.uint8))
print('MSE: ',np.mean(np.power(img-img_con.astype(np.uint8),2)))

# coeff_grid = utils.make_grid_coeff(coeff, normalize=True)
# cv2.imshow('coeff', coeff_grid)
cv2.waitKey(0)



