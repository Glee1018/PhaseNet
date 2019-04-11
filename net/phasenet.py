from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchvision

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
import os.path
import sys

pi = np.pi

def normalize(input_):
    '''
    Nomalize the phase values by dividing by pi.

    The residual and amplitude values are normlized 
    by dividing by the maximum value of the corresponding level.
    '''
    temp = input_
    if input_.shape[1] == 2:  # Normlize residual!!!!
        for i in range(temp.shape[0]):
            for j in range(2):
                temp[i, j, :, :] = temp[i, j, :, :]/temp[i, j, :, :].max()
    else:  # Normlize bands!!!!
        bands = int(input_.shape[1]/4)
        # phase
        for i in range(bands):
            temp[:, i+bands, :, :] = temp[:, i+bands, :, :]/pi
            temp[:, i+3*bands, :, :] = temp[:, i+3*bands, :, :]/pi
        # amplitude
        for i in range(temp.shape[0]):
            for j in range(bands):
                temp[i, j, :, :] = temp[i, j, :, :]/temp[i, j, :, :].max()
                temp[i, j+2*bands, :, :] = temp[i, j+2*bands, :, :] / temp[i, j+2*bands, :, :].max()
    return temp


def get_phase(complex_input):
    '''
    Args: 
        complex_input: type~np.array
                        dtype=np.complex
    Return:
        torch.tensor
    '''
    return torch.from_numpy(np.arctan2(complex_input.imag, complex_input.real))


def get_amplitude(complex_input):
    '''
    Args: 
        complex_input: type~np.array
                       dtype=np.complex
    Return:
        torch.tensor
    '''
    return torch.from_numpy(np.abs(complex_input))


def convert(coeff_start, coeff_inter, coeff_end):
    '''
    train: coeff_start and coeff_end

    truth: coeff_inter

    '''
    coeff_start_inv = coeff_start[::-1]
    coeff_inter_inv = coeff_inter[::-1]
    coeff_end_inv = coeff_end[::-1]
    train = []
    truth = []

    # low level residual
    train.append(torch.stack([torch.from_numpy(np.fft.ifft2(np.fft.ifftshift(coeff_start_inv[0])).real),
                              torch.from_numpy(np.fft.ifft2(np.fft.ifftshift(coeff_end_inv[0])).real)]))
    truth.append(torch.unsqueeze(torch.from_numpy(
        np.fft.ifft2(np.fft.ifftshift(coeff_inter_inv[0])).real), 0))

    # bands
    for i in range(1, len(coeff_start)-1):
        train.append(torch.stack([get_amplitude(coeff_start_inv[i][0]), get_amplitude(coeff_start_inv[i][1]),
                                  get_amplitude(coeff_start_inv[i][2]), get_amplitude(coeff_start_inv[i][3]),
                                  get_phase(coeff_start_inv[i][0]), get_phase(coeff_start_inv[i][1]),
                                  get_phase(coeff_start_inv[i][2]), get_phase(coeff_start_inv[i][3]),
                                  get_amplitude(coeff_end_inv[i][0]), get_amplitude(coeff_end_inv[i][1]),
                                  get_amplitude(coeff_end_inv[i][2]), get_amplitude(coeff_end_inv[i][3]),
                                  get_phase(coeff_end_inv[i][0]), get_phase(coeff_end_inv[i][1]),
                                  get_phase(coeff_end_inv[i][2]), get_phase(coeff_end_inv[i][3])]))
        truth.append(torch.stack([get_amplitude(coeff_inter_inv[i][0]), get_amplitude(coeff_inter_inv[i][1]),
                                  get_amplitude(coeff_inter_inv[i][2]), get_amplitude(coeff_inter_inv[i][3]),
                                  get_phase(coeff_inter_inv[i][0]), get_phase(coeff_inter_inv[i][1]),
                                  get_phase(coeff_inter_inv[i][2]), get_phase(coeff_inter_inv[i][3])]))

    return train, truth


def get_input(batch_coeff):
    res = [convert(item[0], item[1], item[2]) for item in batch_coeff]
    train = []
    truth = []
    for i in range(len(res[0][0])):
        train.append(torch.stack([tt[0][i] for tt in res]).float())
        truth.append(torch.stack([tt[1][i] for tt in res]).float())
    return train, truth


def pil_loader(path):
    # copy from torchvision.datasets.ImageFolder source code
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def show_Triplets_batch(Triplets_batch):
    '''
    Added by Lijie

    Show Triplets_batch

    Args:
        Triplets_batch ({}): a image batch from Triplets

    Return: 
        image batch (numpy)
    '''
    batch_size = len(Triplets_batch['start'])
    img_list = []
    for i in range(batch_size):
        img_list.append(Triplets_batch['start'][i])
        img_list.append(Triplets_batch['inter'][i])
        img_list.append(Triplets_batch['end'][i])
    im_batch = torchvision.utils.make_grid(img_list, nrow=3).numpy()
    im_batch = np.transpose(im_batch, (1, 2, 0))
    plt.imshow(im_batch)
    plt.show()
    return im_batch


class Triplets(Dataset):
    '''
    Added by Lijie

    Generate dataset as it's showed below:

    {'start':[class1_first, class2_first, ...],
     'inter':[class1_inter, class2_inter, ...],
     'end':  [class1_end,class2_end,...],
     'class_index':[class1_index, class2_index, ...]}

    the len of each value in the Dataset dict : batch_size
    '''

    def __init__(self, root_dir, transform=None):
        """
        dir:
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

        Args:
            root_dir (string): Directory with all the images.

        """
        self.root_dir = root_dir
        self.classes, self.class_to_idx = self._find_classes()
        self.sample = self._make_sample()
        self.transform = transform

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        # [(path,index),(path,index),(path,index)]
        imgs_list = self.sample[idx]
        pil_loader(imgs_list[0][0])
        sample = [pil_loader(img[0]) for img in imgs_list]
        if self.transform:
            sample = [self.transform(item) for item in sample]
        sample = {'start': sample[0], 'inter': sample[1],
                  'end': sample[2], 'class_index': imgs_list[0][1]}
        return sample

    def _find_classes(self):
        """
        Copy from torchvision.datasets.ImageFolder source code

        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_sample(self):
        # a modified version from torchvision.datasets.ImageFolder source code
        sample = []
        dir = os.path.expanduser(self.root_dir)
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            images = []
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target])
                    images.append(item)
            for i in range(len(images)-2):
                sample.append(images[i:i+3])
        return sample


# def Resize(input, New_Size):
#     '''
#     Added by Lijie

#     Resize torch.Tensor 's H and W

#     Args:
#         input: torch.Tensor (N,C,H,W)  N:batch_size C:channels
#         New_size: (H_new,W_new)

#     Return torch.Tensor (N,C,H_new,W_new)
#     '''
#     assert isinstance(input, torch.Tensor)
#     input_np = input.numpy()
#     input_size = input_np.shape
#     print(input_size)
#     output = np.empty(
#         shape=(input_size[0], input_size[1], New_Size[0], New_Size[1]))
#     print(output.shape)
#     for i in range(input_size[0]):
#         for j in range(input_size[1]):
#             temp = Image.fromarray(input_np[i, j, :, :])
#             temp = temp.resize(New_Size, Image.BILINEAR)
#             output[i, j, :, :] = np.array(temp)
#     return torch.from_numpy(output)

class PhaseNetBlock(nn.Module):
    # PhaseNetBlock，return feature map
    def __init__(self, in_channels=88, out_channels=64, kernel_size=3, padding=1):
        super(PhaseNetBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.Conv2d(out_channels, out_channels,kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class Pred(nn.Module):
    # Pred，return pred
    def __init__(self, in_channels=64, out_channels=8, kernel_size=1):
        super(Pred, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        out = F.tanh(self.conv(x))
        return out


class PhaseNet(nn.Module):
    '''
    Added by Lijie

    the net proposed in paper "PhaseNet for Video Frame Interpolation"(https://arxiv.org/abs/1804.00884v1)

    input:
        truth_coeff,pre_coeff,truth_img,pre_img

    '''

    def __init__(self):
        super(PhaseNet, self).__init__()
        # alpha and beta are used to predict the low level residual and amplitude values  according to the 'first' and 'end' frame
        self.alpha = torch.nn.Parameter(torch.rand(1))
        self.beta = torch.nn.Parameter(torch.rand(1))

        self.layer = nn.ModuleList()
        self.pred = nn.ModuleList()

        self.layer.append(PhaseNetBlock(2, 64, 1, 0))
        self.pred.append(Pred(64, 1))

        self.layer.append(PhaseNetBlock(81, 64, 1, 0))
        self.pred.append(Pred())

        self.layer.append(PhaseNetBlock(kernel_size=1, padding=0))
        self.pred.append(Pred())

        # layer3
        self.layer.append(PhaseNetBlock())
        self.pred.append(Pred())

        self.layer.append(PhaseNetBlock())
        self.pred.append(Pred())

        self.layer.append(PhaseNetBlock())
        self.pred.append(Pred())

        self.layer.append(PhaseNetBlock())
        self.pred.append(Pred())

        self.layer.append(PhaseNetBlock())
        self.pred.append(Pred())

        self.layer.append(PhaseNetBlock())
        self.pred.append(Pred())

        self.layer.append(PhaseNetBlock())
        self.pred.append(Pred())

        # layer10
        self.layer.append(PhaseNetBlock())
        self.pred.append(Pred())

    def forward(self, x):
        feature_map = []
        pred_map = []
        output = []

        feature_map.append(self.layer[0](normalize(x[0])))
        pred_map.append(self.pred[0](feature_map[0]))
        amp = self.alpha*x[0][:, 0, :, :]+(1-self.alpha)*x[0][:, 1, :, :]
        output.append(torch.unsqueeze(amp,1))

        for i in range(1, len(x)):
            img_shape = (x[i].shape[2], x[i].shape[3])
            feature_map.append(self.layer[i](torch.cat([normalize(x[i]),
                                            F.interpolate(feature_map[i-1], img_shape, mode='bilinear'),
                                            F.interpolate(pred_map[i-1], img_shape, mode='bilinear')], 1)))
            pred_map.append(self.pred[i](feature_map[i]))
            amp = self.beta*x[i][:, 0:4, :, :] + (1-self.beta)*x[i][:, 8:12, :, :]
            phase = pred_map[i][:,4:8,:,:]
            output.append(torch.cat([amp, phase],1))
        return output

class Total_loss(nn.Module):
    '''
    Added by Lijie

    the loss proposed in paper "PhaseNet for Video Frame Interpolation"(https://arxiv.org/abs/1804.00884v1)

    input:
        truth_coeff,pre_coeff,truth_img,pre_img

    '''

    def __init__(self, v=0.1):
        super(Total_loss, self).__init__()
        self.v = v

    def forward(self, truth_coeff, pre_coeff, truth_img, pre_img):
        img_loss = nn.L1Loss()(truth_img, pre_img)
        dphase = [truth_coeff[i][:, 4:, :, :]-pre_coeff[i][:, 4:, :, :]
                  for i in range(1, len(truth_coeff))]
        atan2_phase = [torch.atan2(torch.sin(d), torch.cos(d)) for d in dphase]
        phase_loss = 0
        for i in range(len(dphase)):
            phase_loss += nn.L1Loss()(atan2_phase[i],
                                      torch.zeros_like(atan2_phase[i]))
        return self.v*phase_loss+img_loss


if __name__ == "__main__":
    model = PhaseNet()
    # print(model)
    for i, j in model.named_parameters():
        print(i)

    # # test input
    # input=[]
    # input.append(torch.autograd.Variable(torch.randn(2, 2, 8, 8)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 12, 12)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 16, 16)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 22, 22)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 32, 32)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 46, 46)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 64, 64)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 90, 90)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 128, 128)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 182, 182)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 256, 256)))
    # o = model(input)
