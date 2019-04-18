import torch
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import cv2

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from net.phasenet import PhaseNet, Triplets, show_Triplets_batch, get_input,output_convert

# Load dataset
batch_size = 1
transform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor()])

dataset = Triplets(
    '/home/lj/Documents/code/python/DAVIS/JPEGImages/480p/', transform)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

dataiter = iter(trainloader)
Triplets_batch = dataiter.next()
# show_Triplets_batch(Triplets_batch)
# define para
height = 12
nbands = 4
scale_factor = 2**(1/2)
pyr_type = 1
device = torch.device('cpu')# cuda:0
pyr = SCFpyr_PyTorch(height=height, nbands=nbands, scale_factor=scale_factor,device=device)


# load model
model = torch.load('./model/2019-04-15 21:46:19_model.pkl')
model.eval()
# get images_list [[N,C,H,W],[N,C,H,W],...] len(images_list)=batch_size
images_list = [torch.stack([Triplets_batch['start'][i],
                            Triplets_batch['inter'][i],
                            Triplets_batch['end'][i]]) for i in range(batch_size)]

img_recon=np.empty(shape=(256,256,3))
plt.figure(1)
with torch.no_grad():
    for channel in range(3):
        batch_coeff_list = [pyr.build(image[:,channel,:,:].unsqueeze(1).to(device), pyr_type=pyr_type)
                                for image in images_list]
        train_coeff, truth_coeff = get_input(batch_coeff_list)
        pre_coeff = model(train_coeff)

        truth_img = Triplets_batch['inter'][:, channel, :, :]
        pre_img = pyr.reconstruct(output_convert(pre_coeff),pyr_type=pyr_type)
        print(pre_img.shape)
        # import pdb;pdb.set_trace()
        img = pre_img[0].numpy()
        # img = 255*(img-img.min())/(img.max()-img.min())
        img_recon[:,:,channel] = img
        plt.subplot(1,3,channel+1)
        plt.imshow((255*(img-img.min())/(img.max()-img.min())).astype(np.uint8),'gray')
        plt.title('channel:{}'.format(channel))

plt.figure(2)
plt.subplot(1,2,1)
plt.imshow((img_recon*255).astype(np.uint8))
plt.title('img_recon')
plt.subplot(1,2,2)
truth = Triplets_batch['inter'].squeeze().numpy().transpose(1,2,0)*255
plt.imshow(truth.astype(np.uint8))
plt.title('ground_truth')
plt.show()
