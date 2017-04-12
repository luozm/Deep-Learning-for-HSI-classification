"""Definitions and global settings.

Any changes only need to be done here.
-- File settings
-- Pre-processing settings
-- Model settings

Functions for handling the Hyperspectral data.
-- class DataSet
-- def read_data_sets

@Author: lzm
"""

import os

# File settings

data_path = './data'  # Path of the data
model_path = './model'  # Path of model's checkpoints
graph_path = './graph'  # Path of graphs for TFboard

data_file = 'Indian_pines'  # File name of the data set
data_name = 'indian_pines'  # Matrix name of the data set within the data file
classes = 16  # number of target labels
bands = 200  # number of spectral channels


# Pre-processing settings

patch_size = 11  # Size of the CNN respective field
test_frac = 0.9  # Fraction of data to be used for testing

oversample = True  # Whether use over-sample or just shuffle
patch_class = 200  # Number of training patches of each class (if over-sample)


# model settings

learning_rate = 0.01  # Initial learning rate
epochs = 5  # Number of steps to run trainer
batch_size = 100  # Batch size, must divide evenly into the dataset sizes
model = 'cnn_3d'  # Set the model type
