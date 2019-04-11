import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt
import cv2

from steerable import utils, SCFpyr_NumPy
from steerable.SCFpyr_NumPy import SCFpyr_NumPy
from net.phasenet import PhaseNet,Triplets,show_Triplets_batch,get_input,Total_loss

# Load dataset
batch_size=4
transform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor()])

dataset=Triplets('/home/lj/Documents/code/python/DAVIS/JPEGImages/480p/',transform)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

dataiter = iter(trainloader)
Triplets_batch = dataiter.next()
# show_Triplets_batch(Triplets_batch)

# print(labels)
# print(images.shape)
# utils.show_image_batch(images)
# print(images.max())
# ttt=torchvision.transforms.ToPILImage()(images[0])
# print(np.array(ttt).dtype)
# from PIL import Image
# ttt.show()

# define para
height = 12
nbands = 4
scale_factor = 2**(1/2)
pyr = SCFpyr_NumPy(height=height, nbands=nbands, scale_factor=scale_factor)
import pdb;pdb.set_trace()
# define network
model=PhaseNet()

# get images_list [[N,C,H,W],[N,C,H,W],...] len(images_list)=batch_size
images_list = [torch.stack([Triplets_batch['start'][i],
                            Triplets_batch['inter'][i],
                            Triplets_batch['end'][i]]) for i in range(batch_size)]

# do csp on channel 0
channel=0
# print(images_list[0].shape)
batch_coeff_list = [pyr.BatchCsp(image, channel=channel, type=1) for image in images_list]
train_coeff,truth_coeff=get_input(batch_coeff_list)

pre_coeff=model(train_coeff)
import pdb;pdb.set_trace()
# calculate loss
truth_img=Triplets_batch['inter'][:,channel,:,:]
pre_img=torch.stack([torch.from_numpy(i).float() for i in pyr.phasenet_recon(pre_coeff)])

criterion = Total_loss()
loss=criterion(truth_coeff,pre_coeff,truth_img,pre_img)
print(loss)
# import pdb;pdb.set_trace()



# transform coeff to phasenet input







# import pdb;pdb.set_trace()
# coeffs_batch = pyr.BatchCsp(images_list[0], channel=0, type=1)
# img_recon_batch = pyr.Batch_recon(coeffs_batch, type=1)



# # visualize
# coeff_grid = utils.make_grid_coeff(coeffs_batch[0], normalize=True)
# cv2.imshow('coeff', coeff_grid)
# # cv2.imshow('img_recon', img_con.astype(np.uint8))
# cv2.imshow('img_recon', img_con_batch[0].astype(np.uint8))
# cv2.waitKey(0)
