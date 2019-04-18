import torch
import torchvision
from torchvision import transforms

import time
import os
import numpy as np

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from net.phasenet import PhaseNet, Triplets, show_Triplets_batch, get_input, Total_loss, output_convert

# create log
log_dir = './log/'
if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
    os.makedirs(log_dir)
now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
log_name = '{}_train.txt'.format(now)
device = torch.device('cpu')# cuda:0

# define net parameter
num_epochs = 2
learning_rate = 0.001
batch_size = 8
# pyr parameter
height = 12
nbands = 4
scale_factor = 2**(1/2)
pyr_type = 1
# Load dataset
transform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor()])
dataset = Triplets(
    '/home/lj/Documents/code/python/DAVIS/JPEGImages/480p/', transform)

pyr = SCFpyr_PyTorch(height=height, nbands=nbands, scale_factor=scale_factor, device=device)
# define network
model = PhaseNet()
criterion = Total_loss(v=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999))

# Train the model
total_step = 0
for epoch in range(num_epochs):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
    for channel in range(3):
        for n, Triplets_batch in enumerate(trainloader):
            # get images_list [[N,C,H,W],[N,C,H,W],...], usually len(images_list)=batch_size if len(dataset)%batch_size==0
            images_list = [torch.stack([Triplets_batch['start'][i],
                                        Triplets_batch['inter'][i],
                                        Triplets_batch['end'][i]]) for i in range(len(Triplets_batch['start']))]
            # batch_coeff_list = [pyr.BatchCsp(
            #     image, channel=channel, type=1) for image in images_list]
            batch_coeff_list = [pyr.build(image[:,channel,:,:].unsqueeze(1).to(device), pyr_type=pyr_type)
                                for image in images_list]
            train_coeff, truth_coeff = get_input(batch_coeff_list)

            # Forward pass
            pre_coeff = model(train_coeff)

            truth_img = Triplets_batch['inter'][:, channel, :, :]
            pre_img = pyr.reconstruct(output_convert(pre_coeff),pyr_type=pyr_type)
            # import pdb;pdb.set_trace()
            loss = criterion(truth_coeff, pre_coeff, truth_img, pre_img)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(model.state_dict()['alpha'])
            # print(model.state_dict()['pred.1.conv.bias'])
            total_step+=1
            if total_step % 10 == 0:
                print('Epoch [{}/{}], Channel [{}], Step [{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, channel, total_step, loss.item()))
                # save log
                with open(os.path.join(log_dir, log_name), 'at') as f:
                    f.write('Epoch [{}/{}], Channel [{}], Step [{}], Loss: {:.4f}\n'.format(
                        epoch+1, num_epochs, channel, total_step, loss.item()))
# save model
torch.save(model, './model/{}_model.pkl'.format(now))
