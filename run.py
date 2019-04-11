import pdb
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from steerable import utils, SCFpyr_NumPy
from steerable.SCFpyr_NumPy import SCFpyr_NumPy
from net.phasenet import PhaseNet, Triplets, show_Triplets_batch, get_input, Total_loss

# create log
log_dir = './log/'
if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
    os.makedirs(log_dir)
now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
log_name = '{}_train.txt'.format(now)

# define net parameter
num_epochs = 12
learning_rate = 0.1
batch_size = 4
# pyr parameter
height = 12
nbands = 4
scale_factor = 2**(1/2)

# Load dataset
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
# pdb.set_trace()
pyr = SCFpyr_NumPy(height=height, nbands=nbands, scale_factor=scale_factor)

# define network
model = PhaseNet()
criterion = Total_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(trainloader)
for epoch in range(num_epochs):
    for channel in range(3):
        for n, Triplets_batch in enumerate(trainloader):
            # get images_list [[N,C,H,W],[N,C,H,W],...] len(images_list)=batch_size
            images_list = [torch.stack([Triplets_batch['start'][i],
                                        Triplets_batch['inter'][i],
                                        Triplets_batch['end'][i]]) for i in range(batch_size)]

            # print(images_list[0].shape)
            batch_coeff_list = [pyr.BatchCsp(
                image, channel=channel, type=1) for image in images_list]
            train_coeff, truth_coeff = get_input(batch_coeff_list)

            # Forward pass
            pre_coeff = model(train_coeff)

            truth_img = Triplets_batch['inter'][:, channel, :, :]
            pre_img = torch.stack([torch.from_numpy(i).float()
                                   for i in pyr.phasenet_recon(pre_coeff)])
            loss = criterion(truth_coeff, pre_coeff, truth_img, pre_img)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (n+1) % 2 == 0:
                print('Epoch [{}/{}], Channel {}, Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, channel, n+1, total_step, loss.item()))
                # save log
                with open(os.path.join(log_dir, log_name), 'at') as f:
                    f.write('Epoch [{}/{}], Channel {}, Step [{}/{}], Loss: {:.4f}\n'.format(
                        epoch+1, num_epochs, channel, n+1, total_step, loss.item()))
                        
torch.save(model, './model/{}_model.pkl'.format(now))
