# -*- coding: utf-8 -*-
__author__ = "Alexander Stepanov"
__license__ = "GNU GPLv3"

"""
Contains function needed to create train set ready for UNET Neural Network model (https://arxiv.org/abs/1505.04597) 
from Data Science Bowl 2018 dataset
https://www.kaggle.com/competitions/data-science-bowl-2018/data

Reference tutorial:
author - Sreenivas Bhattiprolu
channel - https://www.youtube.com/@DigitalSreeni
video - https://youtu.be/azM57JuQpQI
"""

import numpy as np
from tqdm import tqdm
import os
from numpy.random import default_rng
import pickle

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

import cv2



IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

TRAIN_PATH = "g:/Datasets/2018_data_science_bowl/stage1_train/"


def create_train_set():
    '''
    Creates train set ready for UNET Neural Network model
    
    Time of reading and converting images
    ( Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz ):
        
    100%|██████████| 670/670 [05:13<00:00,  2.14it/s]
    
    Returns
    -------
    X and Y tensors of train set

    '''
    rng = default_rng(0)
    # dont't read images the socond time if files already exist
    if os.path.exists("./X_train.pickle") and os.path.exists("./Y_train.pickle"):
        X_train_pickle = open('./X_train.pickle', "rb")
        Y_train_pickle = open('./Y_train.pickle', "rb")
        X_train = pickle.load(X_train_pickle)
        Y_train = pickle.load(Y_train_pickle)
        train_file_names = next(os.walk(TRAIN_PATH))[1]
        img_idx = rng.integers(0, len(train_file_names), size = 1)[0]
        plt.title("Image")
        imshow(X_train[img_idx])
        plt.figure()
        plt.title("Mask")
        imshow(Y_train[img_idx])
        X_train_pickle.close()
        Y_train_pickle.close()
        return X_train, Y_train
    else:
        X_train_pickle = open('./X_train.pickle',"wb")
        Y_train_pickle = open('./Y_train.pickle',"wb")
        # get names of folders (the second element in tuple returned by next() function)
        # names of folders are equal to the names of files
        train_file_names = next(os.walk(TRAIN_PATH))[1]
        
        X_train = np.zeros((len(train_file_names), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        Y_train = np.zeros((len(train_file_names), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        
        for n, file_name in tqdm(enumerate(train_file_names), total=len(train_file_names)):               
            path = TRAIN_PATH + file_name   
            img = imread(path + '/images/' + file_name + '.png')[:, :, :IMG_CHANNELS]  
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_train[n] = img
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',  
                                              preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)  
                    
            Y_train[n] = mask
        
        pickle.dump(X_train, X_train_pickle)
        pickle.dump(Y_train, Y_train_pickle)
        
        X_train_pickle.close()
        Y_train_pickle.close()
        
        return X_train, Y_train
    
    
def create_train_set_with_open_cv():
    '''
    Creates train set ready for UNET Neural Network model
    
    Time of reading and converting images
    ( Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz ):
        
    100%|██████████| 670/670 [00:56<00:00, 11.90it/s]
    
    Returns
    -------
    X and Y tensors of train set

    '''
    rng = default_rng(0)
    # dont't read images the socond time if files already exist
    if os.path.exists("./X_train.pickle") and os.path.exists("./Y_train.pickle"):
        X_train_pickle = open('./X_train.pickle',"rb")
        Y_train_pickle = open('./Y_train.pickle',"rb")
        X_train = pickle.load(X_train_pickle)
        Y_train = pickle.load(Y_train_pickle)
        X_train_pickle.close()
        Y_train_pickle.close()
        train_file_names = next(os.walk(TRAIN_PATH))[1]
        img_idx = rng.integers(0, len(train_file_names), size = 1)[0]
        plt.title("Image")
        imshow(X_train[img_idx])
        plt.figure()
        plt.title("Mask")
        imshow(Y_train[img_idx])
        return X_train, Y_train
    else:
        # create files for saving data
        X_train_pickle = open('./X_train.pickle',"wb")
        Y_train_pickle = open('./Y_train.pickle',"wb")
        
        # get names of folders (the second element in tuple returned by next() function)
        # names of folders are equal to the names of files
        train_file_names = next(os.walk(TRAIN_PATH))[1]
        
        X_train = np.zeros((len(train_file_names), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
        Y_train = np.zeros((len(train_file_names), IMG_HEIGHT, IMG_WIDTH, 1), dtype = bool)
        
        for idx, file_name in tqdm(enumerate(train_file_names), total = len(train_file_names)):   
            train_image = cv2.imread(TRAIN_PATH + file_name + '/images/' + file_name + '.png')
            train_image = cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)
            train_image = cv2.resize(train_image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)  
            X_train[idx] = train_image
            
            train_image_mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype = np.uint8)
            # total image mask is the union of all masks inside corresponding directory
            for mask_file_name in next(os.walk(TRAIN_PATH + file_name + '/masks/'))[2]:
                current_mask = cv2.imread(TRAIN_PATH + file_name + '/masks/' + mask_file_name, cv2.IMREAD_GRAYSCALE)
                current_mask = cv2.resize(current_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)                                              
                current_mask = current_mask.astype(np.uint8)
                train_image_mask = cv2.bitwise_or(train_image_mask, current_mask)
            # ndim should be 3
            # equivalent to train_image_mask[:, :, np.newaxis]
            train_image_mask = np.expand_dims(train_image_mask, axis = -1)
            Y_train[idx] = train_image_mask > 0
        
        # plot random image and corresponding mask
        img_idx = rng.integers(0, len(train_file_names), size = 1)[0]
        plt.title("Image")
        imshow(X_train[img_idx])
        plt.figure()
        plt.title("Mask")
        imshow(Y_train[img_idx])
        
        # save data for further use
        pickle.dump(X_train, X_train_pickle)
        pickle.dump(Y_train, Y_train_pickle)
        
        X_train_pickle.close()
        Y_train_pickle.close()
        
        return X_train, Y_train

    
    

