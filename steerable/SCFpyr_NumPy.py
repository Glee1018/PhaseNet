# Modified by Lijie 
# Email：glee1018@buaa.edu.cn
# Date Modified： 2019-03-20 21:54

################################################################################
# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-12-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import factorial
import torch
import torchvision

import steerable.math_utils as math_utils
pointOp = math_utils.pointOp
        
################################################################################

class SCFpyr_NumPy():
    '''
    Modified by Li Jie
    ---------------------------------------------------------------------------------------------
    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid  using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.

    Description of this transform appears in: Portilla & Simoncelli,
    International Journal of Computer Vision, 40(1):49-71, Oct 2000.
    Further information: http://www.cns.nyu.edu/~eero/STEERPYR/

    Modified code from the perceptual repository:
      https://github.com/andreydung/Steerable-filter

    This code looks very similar to the original Matlab code:
      https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m

    Also looks very similar to the original Python code presented here:
      https://github.com/LabForComputationalVision/pyPyrTools/blob/master/pyPyrTools/SCFpyr.py
        
    function:
        build: build csp, return the result after ifft
        reconstruct: input from build

        build_c: build csp, return the dft result
        reconstruct_c: input from build_c

        pyr_info: print coeff information

        BatchCsp: Do batch CSP
        Batch_recon: Reconstruct images batch from coeffs_batch

    '''

    def __init__(self, height=5, nbands=4, scale_factor=2):
        self.nbands  = nbands  # number of orientation bands
        self.height  = height  # including low-pass and high-pass
        self.scale_factor = scale_factor
        
        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize+1), (self.lutsize+2)))/self.lutsize
        # self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi
        self.alpha = np.mod(self.Xcosn + np.pi,2*np.pi) - np.pi



    ################################################################################
    # Construction of Steerable Pyramid

    def build(self, im):
        ''' Decomposes an image into it's complex steerable pyramid.         
        Args:
            im_batch (np.ndarray): single image [H,W]
        
        Returns:
            pyramid: list containing np.ndarray objects storing the pyramid
        '''

        assert len(im.shape) == 2, 'Input im must be grayscale'
        height, width = im.shape

        # Check whether image size is sufficient for number of levels
        if self.height > int(np.floor(np.log2(min(width, height))/np.log2(self.scale_factor)) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))
        
        # Prepare a grid
        log_rad, angle = math_utils.prepare_grid(height, width)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)


        # Shift the zero-frequency component to the center of the spectrum.
        imdft = np.fft.fftshift(np.fft.fft2(im))

        # Low-pass
        lo0dft = imdft * lo0mask

        # Recursive build the steerable pyramid
        coeff = self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height-1, np.array(im.shape))

        # High-pass
        hi0dft = imdft * hi0mask
        hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))
        coeff.insert(0, hi0.real)
        return coeff


    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height, img_dims):
        #modified by Li Jie
        #add muti scale,for example,scale_factor=2**(1/2)
        if height <= 1:

            # Low-pass
            lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
            coeff = [lo0.real]

        else:
            
            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################
            himask = pointOp(log_rad, Yrcos, Xrcos)

            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2)

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
                banddft = np.power(np.complex(0, -1), self.nbands - 1) * lodft * anglemask * himask
                band = np.fft.ifft2(np.fft.ifftshift(banddft))
                orientations.append(band)

            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################
    
            dims=np.array(lodft.shape)
            ctr=np.ceil((dims+0.5)/2)

            lodims=np.round(img_dims/(self.scale_factor**(self.height-height)))
            loctr=np.ceil((lodims+0.5)/2)
            lostart=(ctr-loctr).astype(np.int)
            loend=(lostart+lodims).astype(np.int)

            # Selection
            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle   = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft   = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]

            # Subsampling in frequency domain
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lodft = lomask * lodft
            
            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################

            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height-1,img_dims)
            coeff.insert(0, orientations)

        return coeff

    ############################################################################
    ########################### RECONSTRUCTION #################################
    ############################################################################

    def reconstruct(self, coeff):

        if self.nbands != len(coeff[1]):
            raise Exception("Unmatched number of orientations")
        
        img_dims=np.array(coeff[0].shape)
        height, width = coeff[0].shape
        log_rad, angle = math_utils.prepare_grid(height, width)

        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos  = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)


        tempdft = self._reconstruct_levels(coeff[1:], log_rad, Xrcos, Yrcos, angle, img_dims)

        hidft = np.fft.fftshift(np.fft.fft2(coeff[0]))
        outdft = tempdft * lo0mask + hidft * hi0mask
        # outdft = tempdft * lo0mask


        reconstruction = np.fft.ifft2(np.fft.ifftshift(outdft))
        reconstruction = reconstruction.real

        return reconstruction

    def _reconstruct_levels(self, coeff, log_rad, Xrcos, Yrcos, angle, img_dims):

        if len(coeff) == 1:
            dft = np.fft.fft2(coeff[0])
            dft = np.fft.fftshift(dft)
            return dft

        Xrcos = Xrcos - np.log2(self.scale_factor)
        # print('len coeff:',len(coeff))
        ####################################################################
        ####################### Orientation Residue ########################
        ####################################################################

        himask = pointOp(log_rad, Yrcos, Xrcos)

        lutsize = 1024
        Xcosn = np.pi * np.array(range(-(2*lutsize+1), (lutsize+2)))/lutsize
        order = self.nbands - 1
        const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
        Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

        orientdft = np.zeros(coeff[0][0].shape)

        for b in range(self.nbands):
            anglemask = pointOp(angle, Ycosn, Xcosn + np.pi * b/self.nbands)
            banddft = np.fft.fftshift(np.fft.fft2(coeff[0][b]))
            orientdft = orientdft + np.power(np.complex(0, 1), order) * banddft * anglemask * himask

        ####################################################################
        ########## Lowpass component are upsampled and convoluted ##########
        ####################################################################

        dims=np.array(coeff[0][0].shape)
        ctr=np.ceil((dims+0.5)/2)

        lodims=np.round(img_dims/(self.scale_factor**(self.height-len(coeff))))
        loctr=np.ceil((lodims+0.5)/2)
        lostart=(ctr-loctr).astype(np.int)
        loend=(lostart+lodims).astype(np.int)

        log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))
        lomask = pointOp(log_rad, YIrcos, Xrcos)

        ################################################################################

        # Recursive call for image reconstruction
        nresdft = self._reconstruct_levels(coeff[1:], log_rad, Xrcos, Yrcos, angle, img_dims)

        resdft = np.zeros(dims, 'complex')
        resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

        return resdft + orientdft


    def build_c(self, im):
        ''' Decomposes an image into it's complex steerable pyramid.         
        Args:
            im_batch (np.ndarray): single image [H,W]
        
        Returns:
            pyramid: list containing complex np.ndarray objects storing the pyramid
        '''

        assert len(im.shape) == 2, 'Input im must be grayscale'
        height, width = im.shape

        # Check whether image size is sufficient for number of levels
        if self.height > int(np.floor(np.log2(min(width, height))/np.log2(self.scale_factor)) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))
        
        # Prepare a grid
        log_rad, angle = math_utils.prepare_grid(height, width)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)
        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)


        # Shift the zero-frequency component to the center of the spectrum.
        imdft = np.fft.fftshift(np.fft.fft2(im))

        # Low-pass
        lo0dft = imdft * lo0mask

        # Recursive build the steerable pyramid
        coeff = self._build_levels_c(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height-1, np.array(im.shape))

        # High-pass
        hi0dft = imdft * hi0mask
        # hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))
        # coeff.insert(0, hi0.real)
        coeff.insert(0, hi0dft)
        return coeff


    def _build_levels_c(self, lodft, log_rad, angle, Xrcos, Yrcos, height, img_dims):
        '''
        Modified by Li Jie
        
        Add muti scale,for example,scale_factor=2**(1/2)
        '''
        if height <= 1:
            coeff = [lodft]

        else:
            
            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################
            himask = pointOp(log_rad, Yrcos, Xrcos)

            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2)

            # Loop through all orientation bands
            orientations = []
            for b in range(self.nbands):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
                banddft = np.power(np.complex(0, -1), self.nbands - 1) * lodft * anglemask * himask
                orientations.append(banddft)

            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################

            dims=np.array(lodft.shape)
            ctr=np.ceil((dims+0.5)/2)

            lodims=np.round(img_dims/(self.scale_factor**(self.height-height)))
            loctr=np.ceil((lodims+0.5)/2)
            lostart=(ctr-loctr).astype(np.int)
            loend=(lostart+lodims).astype(np.int)

            # Selection
            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle   = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft   = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]

            # Subsampling in frequency domain
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lodft = lomask * lodft

            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################

            coeff = self._build_levels_c(lodft, log_rad, angle, Xrcos, Yrcos, height-1,img_dims)
            coeff.insert(0, orientations)

        return coeff

    ############################################################################
    ########################### RECONSTRUCTION #################################
    ############################################################################

    def reconstruct_c(self, coeff):

        if self.nbands != len(coeff[1]):
            raise Exception("Unmatched number of orientations")
        
        img_dims=np.array(coeff[0].shape)
        height, width = coeff[0].shape
        log_rad, angle = math_utils.prepare_grid(height, width)

        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos  = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)


        tempdft = self._reconstruct_levels_c(coeff[1:], log_rad, Xrcos, Yrcos, angle, img_dims)

        hidft = coeff[0]
        outdft = tempdft * lo0mask + hidft * hi0mask
        # outdft = tempdft * lo0mask
        reconstruction = np.fft.ifft2(np.fft.ifftshift(outdft))
        reconstruction = reconstruction.real

        return reconstruction

    def _reconstruct_levels_c(self, coeff, log_rad, Xrcos, Yrcos, angle, img_dims):

        if len(coeff) == 1:
            return coeff[0]

        Xrcos = Xrcos - np.log2(self.scale_factor)
        ####################################################################
        ####################### Orientation Residue ########################
        ####################################################################

        himask = pointOp(log_rad, Yrcos, Xrcos)

        lutsize = 1024
        Xcosn = np.pi * np.array(range(-(2*lutsize+1), (lutsize+2)))/lutsize
        order = self.nbands - 1
        const = np.power(2, 2*order) * np.square(factorial(order)) / (self.nbands * factorial(2*order))
        Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

        orientdft = np.zeros(coeff[0][0].shape)
        for b in range(self.nbands):
            anglemask = pointOp(angle, Ycosn, Xcosn + np.pi * b/self.nbands)
            banddft = coeff[0][b]
            orientdft = orientdft + np.power(np.complex(0, 1), order) * banddft * anglemask * himask

        ####################################################################
        ########## Lowpass component are upsampled and convoluted ##########
        ####################################################################

        dims=np.array(coeff[0][0].shape)
        ctr=np.ceil((dims+0.5)/2)

        lodims=np.round(img_dims/(self.scale_factor**(self.height-len(coeff))))
        loctr=np.ceil((lodims+0.5)/2)
        lostart=(ctr-loctr).astype(np.int)
        loend=(lostart+lodims).astype(np.int)

        log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
        angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))
        lomask = pointOp(log_rad, YIrcos, Xrcos)

        ################################################################################

        # Recursive call for image reconstruction
        nresdft = self._reconstruct_levels_c(coeff[1:], log_rad, Xrcos, Yrcos, angle, img_dims)

        resdft = np.zeros(dims, 'complex')
        resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

        return resdft + orientdft

    @staticmethod
    def pyr_info(coeff):
        '''
        Added by Lijie

        Print pyramid info

        Args:
            coeff (list): The image decomposition by complex steerable pyramid
        '''
        assert isinstance(coeff, list)
        height=len(coeff)
        bands=len(coeff[1])
        print('-----------------------------------')
        print('Pyr height: {} Pyr bands: {}'.format(height,bands))
        print('Pyr{0:2d}: {1}  {2}'.format(height,coeff[0].dtype,coeff[0].shape))
        for i in range(height-2):
            print('Pyr{0:2d}: {1}  {2}'.format(height-i-1,coeff[i+1][0].dtype,coeff[i+1][0].shape))
        print('Pyr{0:2d}: {1}  {2}'.format(1,coeff[height-1].dtype,coeff[height-1].shape))
        print('-----------------------------------')

    def BatchCsp(self, imgs_batch, channel, type):
        '''
        Added by Lijie

        Do batch csp

        Args:
            img_batch: images batch[N,C,H,W], usually from torch.utils.data.DataLoader

            channel: do csp on this channel

            type: 0 for build
                  1 for build_c 
        
        Return：
            BatchCsp(np.array) list
        '''
        
        assert isinstance(
            imgs_batch, torch.Tensor), 'imgs_batch must be type torch.Tensor！'
        assert imgs_batch.shape[1] >= channel, 'Invalid input channe！'
        assert type==0 or type==1, 'Invalid input type！'
        coeffs_batch = []
        if type==0:
            for i in range(imgs_batch.shape[0]):
                # img_array = np.array(torchvision.transforms.ToPILImage()(imgs_batch[i, channel, :, :]))
                img_array = imgs_batch[i, channel, :, :].numpy()
                coeffs_batch.append(self.build(img_array))
            return coeffs_batch
        else:
            for i in range(imgs_batch.shape[0]):
                # img_array = np.array(torchvision.transforms.ToPILImage()(imgs_batch[i, channel, :, :]))
                img_array = imgs_batch[i, channel, :, :].numpy()
                coeffs_batch.append(self.build_c(img_array))
            return coeffs_batch
            

    def Batch_recon(self,coeffs_batch,type):
        '''
        Add by Lijie

        Reconstruct images batch from coeffs_batch

        Args：
            coeffs_batch: Result from Funciton BatchCsp
            type: 0 for build
                  1 for build_c 

        Return: 
            image(np.array) batch list
        '''
        assert type==0 or type==1, 'Invalid input type！'
        if type==0:
            return [self.reconstruct(coeff) for coeff in coeffs_batch ]
        if type==1:
            return [self.reconstruct_c(coeff) for coeff in coeffs_batch ]



    # def phasenet_recon(self,output):
    #     '''
    #     Only for PhaseNet output, to reconstruct img

    #     Args:
    #         output: PhaseNet output

    #     Return:
    #         image(np.array) batch list
    #     '''
    #     assert self.nbands == int(output[1].shape[1]/2),'error input!'
    #     coeffs_batch = []
    #     batch_size = output[0].shape[0]            
    #     for i in range(batch_size):
    #         temp=[]
    #         for j in range(len(output)):
    #             if j==0:#insert lodft
    #                 temp.insert(0,np.fft.fftshift(np.fft.fft2(output[j][i].detach().numpy())))
    #             else:
    #                 banddft_list = []
    #                 for n in range(self.nbands):
    #                     bands = np.empty(shape=(output[j].shape[2],output[j].shape[3]),dtype=np.complex)
    #                     amp = output[j][i][n].detach().numpy()
    #                     phase = output[j][i][n+self.nbands].detach().numpy()*np.pi
    #                     bands.real = amp*np.cos(phase)
    #                     bands.imag = amp*np.sin(phase)
    #                     banddft_list.append(bands)
    #                 temp.insert(0,banddft_list)
    #         temp.insert(0,np.zeros(shape=(output[-1].shape[2],output[-1].shape[3]),dtype=np.complex))#insert hidft(zero)
    #         coeffs_batch.append(temp)
    #     return self.Batch_recon(coeffs_batch,1)








