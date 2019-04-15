import torch
import torchvision
from torchvision import transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils
from net.phasenet import Triplets,show_Triplets_batch
# Load dataset
batch_size = 4
transform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor()])

dataset = Triplets(
    '/home/lj/Documents/code/python/DAVIS/JPEGImages/480p/', transform)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

dataiter = iter(trainloader)
Triplets_batch = dataiter.next()
# get images_list [[N,C,H,W],[N,C,H,W],...], usually len(images_list)=batch_size if len(dataset)%batch_size==0
images_list = [torch.stack([Triplets_batch['start'][i],
                                        Triplets_batch['inter'][i],
                                        Triplets_batch['end'][i]]) for i in range(len(Triplets_batch['start']))]
# show_Triplets_batch(Triplets_batch)

# Requires PyTorch with MKL when setting to 'cpu' 
device = torch.device('cuda:0')

# Load batch of images [N,1,H,W]
# im_batch_numpy = utils.load_image_batch('./assets/lena.jpg',32,200)
# im_batch_torch = torch.from_numpy(im_batch_numpy).to(device)


# Initialize Complex Steerbale Pyramid
height = 12
nbands = 4
scale_factor = 2**(1/2)
pyr = SCFpyr_PyTorch(height=height, nbands=nbands, scale_factor=scale_factor, device=device)
pyr_type = 1
# Decompose entire batch of images 
coeff = pyr.build(images_list[0][:,0,:,:].unsqueeze(1).to(device), pyr_type=pyr_type)
import pdb;pdb.set_trace()
# Reconstruct batch of images again
batch_recon = pyr.reconstruct(coeff,pyr_type=pyr_type)
cv2.imshow('img_recon',(batch_recon[0].cpu().numpy()*255).astype(np.uint8))
# # Visualization
# coeff_single = utils.extract_from_batch(coeff, 0)
# coeff_grid = utils.make_grid_coeff(coeff_single, normalize=True)
# cv2.imshow('Complex Steerable Pyramid', coeff_grid)
cv2.waitKey(0)