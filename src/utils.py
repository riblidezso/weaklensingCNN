#!/usr/bin/env python
# coding: utf-8
# *Author: Dezso Ribli*

"""

Util functions for training CNN on weak lesnsing maps.
Mostly data loaders and data generators with some
additional functionality.

"""

import numpy as np
# https://github.com/IntelPython/mkl_fft/issues/11
#np.fft.restore_all()

import cv2
import math
import os

def step_decay(epoch, base_lr, epochs_drop, drop=0.1):
    """Helper for step learning rate decay."""
    lrate = base_lr
    for epoch_drop in epochs_drop:
        lrate *= math.pow(drop,math.floor(epoch/epoch_drop))
        return lrate


def load_training_data(mapsize=512, grfized=False, exclude_fid=False,
                       dense_grid=False, random_split=False, 
                       from_files=False):
    """Load data for different training scenarios."""
    if not grfized and (not dense_grid) and (not random_split): 
        # the default data to loas
        X_train, X_test, y_train, y_test = load_sparse_grid(imsize=mapsize,
                from_files=from_files)
    elif grfized:
        # equivalent gaussian random filed maps
        assert not from_files
        X_train, X_test, y_train, y_test = load_grf_sparse_grid()
    elif dense_grid:
        assert not from_files
        # data with additional points around a cosmology
        X_train, X_test, y_train, y_test = load_dense_grid(imsize=mapsize)
    elif random_split:
        # random train and test split
        X_train, X_test, y_train, y_test = load_randomsplit_grid(
            imsize=mapsize, from_files=from_files)
  
    # aleays predict newidf, why not, it takes not time 
    # anyway we will not use it with the experiemnts
    fn = '../../data/columbia_data_fiducial_new_idf_pix'+str(mapsize)+'.npy'
    X_new_idf = np.load(fn)
    y_new_idf = np.ones((len(y_test),2))
    y_new_idf[:,0], y_new_idf[:,1] = 0.309, 0.816

    if exclude_fid:  # exclude fiducial cosmo params if asked for
        idx = (y_train[:,0] == 0.309) &  (y_train[:,1] == 0.816)
        X_train, y_train = X_train[~idx], y_train[~idx]
        
    return X_train, X_test, X_new_idf, y_train, y_test, y_new_idf 

    
def load_sparse_grid(d='../../data/sparsegrid/', imsize = 512, 
                     from_files=False):
    if from_files:  # only load filenames
        X_train = np.arange(len(os.listdir(os.path.join(d, 'train'))))
        X_test  = np.arange(len(os.listdir(os.path.join(d, 'test'))))
    else:  # load the files themselves
        X_train = np.load(d+'sparse_grid_final_'+str(imsize)+'pix_x_train.npy')
        X_test =  np.load(d+'sparse_grid_final_'+str(imsize)+'pix_x_test.npy')
    y_train = np.load(d+'sparse_grid_final_'+str(imsize)+'pix_y_train.npy')
    y_test =  np.load(d+'sparse_grid_final_'+str(imsize)+'pix_y_test.npy')
    return X_train, X_test, y_train, y_test


"""Loaders for various experiments."""

def load_grf_sparse_grid(d='../../data/grf/', case='a',imsize=512):
    X_train = np.load(d+'grf'+case+'_sparse_grid_final_'+str(imsize)+'pix_x_train.npy')
    X_test =  np.load(d+'grf'+case+'_sparse_grid_final_'+str(imsize)+'pix_x_test.npy')
    y_train = np.load(d+'grf_sparse_grid_final_'+str(imsize)+'pix_y_train.npy')
    y_test =  np.load(d+'grf_sparse_grid_final_'+str(imsize)+'pix_y_test.npy')
    return X_train, X_test, y_train, y_test


def load_dense_grid(d='../../data/densegrid/', imsize = 512):
    X_train = np.load(d+'dense_grid_final_'+str(imsize)+'pix_x_train.npy')
    X_test =  np.load(d+'dense_grid_final_'+str(imsize)+'pix_x_test.npy')
    y_train = np.load(d+'dense_grid_final_'+str(imsize)+'pix_y_train.npy')
    y_test =  np.load(d+'dense_grid_final_'+str(imsize)+'pix_y_test.npy')
    return X_train, X_test, y_train, y_test


def load_randomsplit_grid(d='../../data/randomsplit/sparse_512/', imsize = 512,
                          from_files=False):
    if from_files:  # only load filenames
        X_train = np.arange(len(os.listdir(os.path.join(d, 'train'))))
        X_test  = np.arange(len(os.listdir(os.path.join(d, 'test'))))
    else:  # load the files themselves
        X_train = np.load(d+'sparse_randomsplit_'+str(imsize)+'pix_x_train.npy')
        X_test =  np.load(d+'sparse_randomsplit_'+str(imsize)+'pix_x_test.npy')
    y_train = np.load(d+'sparse_randomsplit_'+str(imsize)+'pix_y_train.npy')
    y_test =  np.load(d+'sparse_randomsplit_'+str(imsize)+'pix_y_test.npy')
    return X_train, X_test, y_train, y_test



    
class DataGenerator():
    """
    Data generator.
    
    Generates minibatches of data and labels.
    
    Usage:
    
    from imgen import ImageGenerator
    g = DataGenerator(data, labels)
    """
    def __init__(self, x, y, batch_size=1, shuffle=True, seed=0,
                 ng=None, smoothing = None, map_size = 512, 
                 y_shape = (2,), augment = False, scale = 60*3.5,
                 d = None, from_files=False):
        """Initialize data generator."""
        self.x, self.y = x, y
        self.from_files = from_files
        self.d = d
        self.batch_size = batch_size
        self.x_shape, self.y_shape = (map_size, map_size, 1), y_shape
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        
        if not from_files:
            assert x.shape[1] == x.shape[2]  # rectangular!!!
        self.A_pix = (float(scale)/map_size)**2
        self.ng = ng
        self.smoothing = smoothing
        self.scale = float(scale)
        
        self.n_data = len(x)
        self.n_steps = len(x)//batch_size  +  (len(x) % batch_size > 0)
        self.i = 0
        self.reset_indices_and_reshuffle(force=True)
        
        
    def reset_indices_and_reshuffle(self, force=False):
        """Reset indices and reshuffle images when needed."""
        if self.i == self.n_data or force:
            if self.shuffle:
                self.index = self.rng.permutation(self.n_data)
            else:
                self.index = np.arange(self.n_data)
            self.i = 0
            
                
    def next(self):
        """Get next batch of images."""
        x = np.zeros((self.batch_size,)+self.x_shape)
        y = np.zeros((self.batch_size,)+self.y_shape)
        for i in range(self.batch_size):
                x[i],y[i] = self.next_one()
        return x,y
    
    
    def next_one(self):
        """Get next 1 image."""
        # reset index, reshuffle if necessary
        self.reset_indices_and_reshuffle()  
        # get next x
        if not self.from_files:  # simply index from array
            x = self.x[self.index[self.i]]
        else:  # load from file
            fn = str(self.x[self.index[self.i]])  + '.npy'
            x = np.load(os.path.join(self.d, fn))
        x = self.process_map(x)
        y = self.y[[self.index[self.i]]]
        self.i += 1  # increment counter
        return x, y
    
    
    def process_map(self, x_in):
        """Process data."""
        x = np.array([x_in],copy=True)            
                
        if self.augment:  # flip and transpose
            x = aug_ims(x, self.rng.rand()>0.5, self.rng.rand()>0.5,
                        self.rng.rand()>0.5)
                
        if self.ng:  # add noise if ng is not None
            x = add_shape_noise(x, self.A_pix, self.ng, self.rng)
            
        if self.smoothing:  # smooth if smoothing is not None
            x[0,:,:,0] = smooth(x[0,:,:,0], self.smoothing, self.scale)
            
        return x
    
    
def predict_on_generator(model, datagen, augment):
    """Predict on data generator with augmentation."""
    datagen.reset_indices_and_reshuffle(force=True)
    y_true, y_pred = [],[]
    for i in range(datagen.n_data):
        xi,yi = datagen.next()
        y_true.append(yi)
        y_pred_tmp = np.zeros(yi.shape)
        if augment:
            for ai in [0,1]:
                for aj in [0,1]:
                    for ak in [0,1]:
                        y_pred_tmp += model.predict_on_batch(
                            aug_ims(xi,ai,aj,ak)) 
            y_pred.append(y_pred_tmp/8.)
        else:
            y_pred.append(model.predict_on_batch(xi))
            
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    return y_true, y_pred


def aug_ims(ims, fliplr=0, flipud=0, T=0):
    """Augment images with flips and transposition."""
    ims_aug = np.array(ims, copy=True)
    for i in range(len(ims_aug)):
        if fliplr:  # flip left right
            ims_aug[i] = np.fliplr(ims_aug[i])
        if flipud:  # flip up down
            ims_aug[i] = np.flipud(ims_aug[i])
        if T:  # transpose
            ims_aug[i,:,:,0] = ims_aug[i,:,:,0].T
    return ims_aug


def add_shape_noise(x, A, ng, rng=None, sige=0.4):
    """Add shape noise"""
    sigpix = sige / (2 * A * ng)**0.5  # final pixel noise scatter
    # add shape noise to map
    if rng:  # use given random generator
        return x + rng.normal(loc=0, scale=sigpix, size=x.shape)
    else:  # or just a random noise
        return x + np.random.normal(loc=0, scale=sigpix, size=x.shape)
    
    
def smooth(x, smoothing_scale_arcmin, map_size_arcmin):
    """Smooth by Gaussian kernel."""
    # smoothing kernel width in pixels instead of arcmins
    map_size_pix = x.shape[0]
    s = (smoothing_scale_arcmin * map_size_pix) / map_size_arcmin
    # cut off at: 6 sigma + 1 pixel
    # for large smooothing area and odd pixel number
    cutoff = 6 * int(s+1) + 1
    return cv2.GaussianBlur(x, ksize=(cutoff, cutoff), sigmaX=s, sigmaY=s)
