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
import tensorflow as tf
import numpy as np
import h5py
import os

# File settings

data_path = os.path.join(os.getcwd(), 'Data')  # Path of the data
model_path = os.path.join(os.getcwd(), 'Model')  # Path of model's checkpoints
graph_path = 'Graph'  # Path of graphs for TFboard

data_file = 'Indian_pines'  # File name of the data set
data_name = 'indian_pines'  # Matrix name of the data set within the data file
classes = 16  # number of target labels
bands = 220  # number of spectral channels


# Pre-processing settings

patch_size = 11  # Size of the CNN respective field
test_frac = 0.5  # Fraction of data to be used for testing

oversample = True  # Whether use over-sample or just shuffle
patch_class = 500  # Number of training patches of each class (if over-sample)


# model settings

learning_rate = 0.01  # Initial learning rate
max_steps = 50000  # Number of steps to run trainer
batch_size = 50  # Batch size, must divide evenly into the dataset sizes


# 2D model

# Convolutional layer 1
filters_conv1 = 200  # Number of filters
kernel_conv1 = [3, 3]  # Kernel size
stride_conv1 = [1, 1]
# Max pool layer 1
pool_size1 = [2, 2]
# Convolutional layer 2
filters_conv2 = 100  # Number of filters
kernel_conv2 = [3, 3]  # Kernel size
stride_conv2 = [1, 1]
# Max pool layer 2
pool_size2 = [2, 2]
# fully connected layers
fc1 = 200  # Number of units in hidden layer 1
drop = 0.5  # dropout rate
fc2 = 84  # Number of units in hidden layer 2


# Define class

class DataSet(object):
    """DataSet class

    Construct a DataSet class for training.

    Attributes (private):
    -- _num_examples: number of examples
    -- _images: input images in the data set
    -- _labels: labels in the data set
    -- _index_in_epoch: index of current batch in epoch
    -- _epochs_completed: number of completed epochs

    Functions:
    -- __init__: initialize DataSet
    -- next_batch: return the next `batch` examples from this data set

    Functions (read only):
    -- images
    -- labels
    -- num_examples
    -- epochs_completed

    """
    def __init__(self, images, labels, dtype=tf.float32):
        """Construct a DataSet.

        `dtype` can be either `uint8` or `float32`.

        FIXME: fake_data options.
        one_hot arg is used only if fake_data is true.

        """
        # Convert the shape from [num_example,channels, height, width]
        # to [num_example, height, width, channels]
        images = np.transpose(images, (0, 2, 3, 1))

        labels = np.transpose(labels)

        dtype = tf.as_dtype(dtype).base_dtype

        # Check errors
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def start_new_epoch(self):
        """
        Start a new epoch.
        Only use for evaluation.

        """
        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        self._index_in_epoch = 0

    def next_batch(self, batch):
        """
        Return the next `batch` examples from this data set.

        Changes:
        -- _index_in_epoch
        -- _epochs_completed
        -- Shuffle the data

        Output:
        -- example's image arrays
        -- example's labels

        """
        start = self._index_in_epoch
        self._index_in_epoch += batch
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch
            assert batch <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], np.reshape(self._labels[start:end], len(self._labels[start:end]))


# Define function

def read_data_sets(directory, value, dtype=tf.float32):
    """
    Load data from HDF5 files.

    Inputs:
    -- directory: file name
    -- value: array name in the file
    -- dtype: DataSet's data type

    Output:
    -- data_sets: loaded DataSet
    """
    with h5py.File(directory, 'r') as readfile:
        images = readfile[value+'_patch'][:]
        labels = readfile[value+'_labels'][:]

    data_sets = DataSet(images, labels, dtype=dtype)

    return data_sets
