from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import numpy as np

import os
import os.path
import sys

import matplotlib.pyplot as plt
def get_phase(complex_input):
    '''
    Args: 
        complex_input: type~np.array
                        dtype=np.complex
    Return:
        torch.tensor
    '''
    return torch.from_numpy(np.arctan2(complex_input.imag,complex_input.real)/np.pi)

def get_amplitude(complex_input):
    '''
    Args: 
        complex_input: type~np.array
                       dtype=np.complex
    Return:
        torch.tensor
    '''
    return torch.from_numpy(np.abs(complex_input))

def convert(coeff_start,coeff_inter,coeff_end):
    coeff_start_inv=coeff_start[::-1]
    coeff_inter_inv=coeff_inter[::-1]
    coeff_end_inv=coeff_end[::-1]
    train=[]
    truth=[]
    # low level residual
    train.append(torch.stack([torch.from_numpy(np.fft.ifft2(np.fft.ifftshift(coeff_start_inv[0])).real),
                              torch.from_numpy(np.fft.ifft2(np.fft.ifftshift(coeff_end_inv[0])).real)]))
    truth.append(torch.unsqueeze(torch.from_numpy(np.fft.ifft2(np.fft.ifftshift(coeff_inter_inv[0])).real),0))
    # band
    for i in range(1,len(coeff_start)-1):
        train.append(torch.stack([get_amplitude(coeff_start_inv[i][0]),get_amplitude(coeff_start_inv[i][1]),
                                get_amplitude(coeff_start_inv[i][2]),get_amplitude(coeff_start_inv[i][3]),
                                get_phase(coeff_start_inv[i][0]),get_phase(coeff_start_inv[i][1]),
                                get_phase(coeff_start_inv[i][2]),get_phase(coeff_start_inv[i][3]),
                                get_amplitude(coeff_end_inv[i][0]),get_amplitude(coeff_end_inv[i][1]),
                                get_amplitude(coeff_end_inv[i][2]),get_amplitude(coeff_end_inv[i][3]),
                                get_phase(coeff_end_inv[i][0]),get_phase(coeff_end_inv[i][1]),
                                get_phase(coeff_end_inv[i][2]),get_phase(coeff_end_inv[i][3])]))
        truth.append(torch.stack([get_amplitude(coeff_inter_inv[i][0]),get_amplitude(coeff_inter_inv[i][1]),
                                get_amplitude(coeff_inter_inv[i][2]),get_amplitude(coeff_inter_inv[i][3]),
                                get_phase(coeff_inter_inv[i][0]),get_phase(coeff_inter_inv[i][1]),
                                get_phase(coeff_inter_inv[i][2]),get_phase(coeff_inter_inv[i][3])]))

    return train,truth

def get_input(batch_coeff):
    res=[convert(item[0],item[1],item[2]) for item in batch_coeff]
    train=[]
    truth=[]
    for i in range(len(res[0][0])):
        train.append(torch.stack([tt[0][i] for tt in res]).float())
        truth.append(torch.stack([tt[1][i] for tt in res]).float())
    return train,truth





def pil_loader(path):
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
    batch_size=len(Triplets_batch['start'])
    img_list=[]
    for i in range(batch_size):
        img_list.append(Triplets_batch['start'][i])
        img_list.append(Triplets_batch['inter'][i])
        img_list.append(Triplets_batch['end'][i])
    im_batch = torchvision.utils.make_grid(img_list,nrow=3).numpy()
    im_batch = np.transpose(im_batch, (1,2,0))
    plt.imshow(im_batch)
    # # plt.axis('off')
    # # plt.tight_layout()
    plt.show()
    return im_batch

class Triplets(Dataset):
    '''
    Add by Lijie

    Generate dataset as it's showed below:

    {'start':[class1_first, class2_first, ...]],
     'inter':[class1_inter, class2_inter, ...]],
     'end':[class1_end,class2_end,...]],
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
        self.classes,self.class_to_idx = self._find_classes()
        self.sample = self._make_sample()
        self.transform = transform

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        imgs_list = self.sample[idx]#[(path,index),(path,index),(path,index)]
        pil_loader(imgs_list[0][0])
        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        sample = [pil_loader(img[0]) for img in imgs_list]
        if self.transform:
            sample = [self.transform(item) for item in sample]
        sample={'start':sample[0],'inter':sample[1],'end':sample[2],'class_index':imgs_list[0][1]}
        return sample

    def _find_classes(self):
        """
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
    # 实现子模块PhaseNetBlock，返回特征图
    def __init__(self, in_channels=2, out_channels=64, kernel_size=1, stride=1):
        super(PhaseNetBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.Conv2d(out_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Pred(nn.Module):
    # 实现子模块Pred，返回预测图
    def __init__(self, in_channels=64, out_channels=8, kernel_size=1, stride=1):
        super(Pred, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        out = torch.tanh(self.conv(x))
        return out




class PhaseNet(nn.Module):
    # 实现主module:PhaseNet
    # PhaseNet包含多个layer
    def __init__(self):
        super(PhaseNet, self).__init__()
        # new
        self.layer=[]
        self.pred=[]
        # layer0
        self.layer.append(PhaseNetBlock(2, 64, 1))
        # print(self.layer[0])
        self.pred.append(Pred(64, 1, 1))


        self.layer.append(PhaseNetBlock(81, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        # layer2
        self.layer.append(PhaseNetBlock(88, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        # layer3
        self.layer.append(PhaseNetBlock(88, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        self.layer.append(PhaseNetBlock(88, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        self.layer.append(PhaseNetBlock(88, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        self.layer.append(PhaseNetBlock(88, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        self.layer.append(PhaseNetBlock(88, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        self.layer.append(PhaseNetBlock(88, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        self.layer.append(PhaseNetBlock(88, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        # layer10
        self.layer.append(PhaseNetBlock(88, 64, 1))
        self.pred.append(Pred(64, 8, 1))

        # old
        # self.layer0 = PhaseNetBlock(2, 64, 1)
        # self.pred0 = Pred(64, 1, 1)

        # self.layer1 = PhaseNetBlock(81, 64, 1)
        # self.pred1 = Pred(64, 8, 1)

        # self.layer2 = PhaseNetBlock(81, 64, 1)
        # self.pred2 = Pred(64, 8, 1)

        # self.layer3 = PhaseNetBlock(81, 64, 3)
        # self.pred3 = Pred(64, 8, 1)

        # self.layer4 = PhaseNetBlock(81, 64, 3)
        # self.pred4 = Pred(64, 8, 1)

        # self.layer5 = PhaseNetBlock(81, 64, 3)
        # self.pred5 = Pred(64, 8, 1)

        # self.layer6 = PhaseNetBlock(81, 64, 3)
        # self.pred6 = Pred(64, 8, 1)

        # self.layer7 = PhaseNetBlock(81, 64, 3)
        # self.pred7 = Pred(64, 8, 1)

        # self.layer8 = PhaseNetBlock(81, 64, 3)
        # self.pred8 = Pred(64, 8, 1)

        # self.layer9 = PhaseNetBlock(81, 64, 3)
        # self.pred9 = Pred(64, 8, 1)

        # self.layer10 = PhaseNetBlock(81, 64, 3)
        # self.pred10 = Pred(64, 8, 1)
 

    def forward(self, x):
        layer=[]
        pred=[]
        layer.append(self.layer[0](x[0]))
        pred.append(self.pred[0](layer[0]))
        for i in range(1,len(x)):
            # print(i)
            img_shape=(x[i].shape[2],x[i].shape[3])            
            layer.append(self.layer[i](torch.cat([x[i],F.interpolate(layer[i-1],img_shape,mode='bilinear'),F.interpolate(pred[i-1],img_shape,mode='bilinear')],1)))
            pred.append(self.pred[i](layer[i]))


        # img_shape=(x[1].shape[2],x[1].shape[3])
        # layer1 = self.layer1(torch.cat([x[1],F.interpolate(layer0,img_shape,mode='bilinear'),F.interpolate(pred0,img_shape,mode='bilinear')]))
        # pred1 = self.pred1(layer0)

        # img_shape=(x[2].shape[2],x[2].shape[3])
        # layer2 = self.layer2(torch.cat([x[2],F.interpolate(layer1,img_shape,mode='bilinear'),F.interpolate(pred1,img_shape,mode='bilinear')]))
        # pred2 = self.pred2(layer2)

        # img_shape=(x[2].shape[2],x[2].shape[3])
        # layer2 = self.layer2(torch.cat([x[2],F.interpolate(layer1,img_shape,mode='bilinear'),F.interpolate(pred1,img_shape,mode='bilinear')]))
        # pred2 = self.pred2(layer2)

        # img_shape=(x[2].shape[2],x[2].shape[3])
        # layer2 = self.layer2(torch.cat([x[2],F.interpolate(layer1,img_shape,mode='bilinear'),F.interpolate(pred1,img_shape,mode='bilinear')]))
        # pred2 = self.pred2(layer2)

        return pred

    # for test
    # def forward(self, x):
    #     layer0 = self.layer0(x)
    #     pred0 = self.pred0(layer0)

    #     # torchvision.transforms.Resize()
    #     # layer1 = self.layer1(x[1])
    #     # pred1 = self.pred1(layer0)

    #     return pred0
class Total_loss(nn.Module):
    '''
        input:
            truth_coeff,pre_coeff,truth_img,pre_img
    '''
    def __init__(self,v=0.1):
        super(Total_loss,self).__init__()
        self.v = v
    def forward(self,truth_coeff,pre_coeff,truth_img,pre_img):
        img_loss = nn.L1Loss()(truth_img,pre_img)
        dphase=[truth_coeff[i][:,4:,:,:]-pre_coeff[i][:,4:,:,:] for i in range(1,len(truth_coeff))]
        print(len(dphase))
        atan2_phase = [torch.atan2(torch.sin(d),torch.cos(d)) for d in dphase]
        phase_loss = 0
        for i in range(len(dphase)):
            phase_loss += nn.L1Loss()(atan2_phase[i],torch.zeros_like(atan2_phase[i]))
            print(phase_loss)
        print(phase_loss,img_loss)
        return self.v*phase_loss+img_loss


if __name__ == "__main__":
    model = PhaseNet()
    

    # test input
    input=[]
    input.append(torch.autograd.Variable(torch.randn(2, 2, 8, 8)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 12, 12)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 16, 16)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 22, 22)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 32, 32)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 46, 46)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 64, 64)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 90, 90)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 128, 128)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 182, 182)))
    input.append(torch.autograd.Variable(torch.randn(2, 16, 256, 256)))
    
    # input = torch.autograd.Variable(torch.randn(2, 2, 8, 8))
    # print(input.shape)
    o = model(input)
    import pdb;pdb.set_trace()
    # o1=torch.nn.functional.interpolate(o,(11,11),mode='bilinear')
    # print(model)
    # print(o.size())
    # print(o1.size())

    # a = torch.autograd.Variable(torch.randn(1, 1, 1, 1))
    # b = torch.autograd.Variable(torch.randn(1, 1, 1, 1))
    # bb = torch.autograd.Variable(torch.randn(1, 1, 1, 1))
    # cc = torch.autograd.Variable(torch.randn(6, 6))
    
    # c = torch.cat([a,bb,b],1)
    # print(c.shape[1])
    # o1=torch.nn.functional.interpolate(c,cc.shape,mode='bilinear')
    # print(o1.shape)
    # print(a)
    # print(bb)
    # print(b)
    # print(c)


