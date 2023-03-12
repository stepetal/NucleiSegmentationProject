# -*- coding: utf-8 -*-
__author__ = "Alexander Stepanov"
__license__ = "GNU GPLv3"

"""
Contains usage of UNET Neural Network model (https://arxiv.org/abs/1505.04597) 
Dataset for training and inference: Data Science Bowl 2018 dataset
                                    https://www.kaggle.com/competitions/data-science-bowl-2018/data

Reference tutorial:
author - Sreenivas Bhattiprolu
channel - https://www.youtube.com/@DigitalSreeni
video - https://youtu.be/azM57JuQpQI

"""

from unet_model import nuclei_nn_model
from prepare_sets import create_train_set_with_open_cv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.io import imshow
from numba import cuda
import tensorflow as tf
import os
import numpy as np
from numpy.random import default_rng

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# get train set (images and masks are processed with opencv)
X, Y = create_train_set_with_open_cv()

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1, random_state = 0)

nuclei_unet_model = nuclei_nn_model(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

# training is not necessary if weights already had been computed
if os.path.exists('./model_for_nuclei.h5'):
    nuclei_unet_model.load_weights('./model_for_nuclei.h5')
    print("weights for model_for_nuclei.h5 were loaded successfully")
else:
    callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
                tf.keras.callbacks.TensorBoard(log_dir='logs'),
                tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose = True, save_best_only=True)
            ]
    
    results = nuclei_unet_model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=30, callbacks=callbacks)

Y_pred = nuclei_unet_model.predict(X_val)

# show some actual and predicted images
rng = default_rng(0)
img_indices = rng.integers(0, X_val.shape[0], size = 8)
nuclei_plot_pred = plt.figure(figsize = (14, 8))
rows_n = 2
cols_n = 4
for subplot_number, img_idx in enumerate(img_indices):
    nuclei_plot_pred.add_subplot(rows_n, cols_n, subplot_number + 1)
    pred_img = (Y_pred[img_idx, :, :, 0] > 0.5).astype(np.uint8)
    plt.imshow(pred_img, cmap = 'gray')
    plt.title("Predicted image idx = {}".format(img_idx))
    
nuclei_plot_actual = plt.figure(figsize = (14, 8))
rows_n = 2
cols_n = 4
for subplot_number, img_idx in enumerate(img_indices):
    nuclei_plot_actual.add_subplot(rows_n, cols_n, subplot_number + 1)
    act_img = Y_val[img_idx].squeeze()
    plt.imshow(act_img, cmap = 'gray')
    plt.title("Actual image idx = {}".format(img_idx))

plt.show()

# clear GPU memory if training was done with GPU (never tried in Google Colab, maybe not necessary there)
if len(tf.config.list_physical_devices('GPU')) > 0:
    cuda.select_device(0)
    cuda.close()




