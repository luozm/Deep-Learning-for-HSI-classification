# -*- coding: utf-8 -*-
"""Preprocessing data.
Load the HSI data sets and split into several patches for CNN.
@Author: lzm
"""

import numpy as np
import scipy.io as scio
from random import shuffle
import Utils
import os
import h5py
import tensorflow as tf


# Define functions

def patch(height_index, width_index):
    """
    Returns a mean-normalized patch, the top left corner of which
    is at (height_index, width_index)
    Inputs:
    -- height_index: row index of the top left corner of the image patch
    -- width_index: column index of the top left corner of the image patch
    Outputs:
    -- mean_normalized_patch: mean normalized patch of size (BAND, PATCH_SIZE, PATCH_SIZE)
    whose top left corner is at (height_index, width_index)
    """
    transpose_array = np.transpose(input_mat, (2, 0, 1))
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patches = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patches.shape[0]):
        mean_normalized_patch.append(patches[i] - MEAN_ARRAY[i])

    return np.array(mean_normalized_patch)


def patch_margin(height_index, width_index):
    """Collect marginal patches

    Returns a mean-normalized patch, the center of which
    is at (height_index, width_index)
    Inputs:
    -- height_index: row index of the center of the image patch
    -- width_index: column index of the center of the image patch
    Outputs:
    -- mean_normalized_patch: mean normalized patch of size (BAND, PATCH_SIZE, PATCH_SIZE)
    whose top left corner is at (height_index, width_index)
    """

    input_mirror = np.zeros(((HEIGHT + PATCH_SIZE - 1), (WIDTH + PATCH_SIZE - 1), BAND))
    # extend the margin of the origin image
    input_mirror[PATCH_IDX:(HEIGHT+PATCH_IDX), PATCH_IDX:(WIDTH+PATCH_IDX), :] = input_mat[:]
    input_mirror[PATCH_IDX:(HEIGHT+PATCH_IDX), :PATCH_IDX, :] = input_mat[:, PATCH_IDX-1::-1, :]
    input_mirror[PATCH_IDX:(HEIGHT+PATCH_IDX), (WIDTH+PATCH_IDX):, :] = input_mat[:, :(WIDTH-PATCH_IDX-1):-1, :]
    input_mirror[:PATCH_IDX, :, :] = input_mirror[(PATCH_IDX*2-1):(PATCH_IDX-1):-1, :, :]
    input_mirror[(HEIGHT+PATCH_IDX):, :, :] = input_mirror[(HEIGHT+PATCH_IDX-1):(HEIGHT-1):-1, :, :]
    transpose_array = np.transpose(input_mirror, (2, 0, 1))

    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patches = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patches.shape[0]):
        mean_normalized_patch.append(patches[i] - MEAN_ARRAY[i])

    return np.array(mean_normalized_patch)


def oversample(truth, train_patch, train_labels, count):
    """
    Over-sample the classes which do not have at least
    COUNT patches in the training set and extract COUNT patches
    Inputs:
    -- truth: whether use over-sample or just shuffle
    -- train_patch: the training set
    -- count: num of samples after over-sample
    Output:
    -- train_patch: Oversampled training set
    """
    if truth:
        for i in range(OUTPUT_CLASSES):
            if len(train_patch[i]) < count:
                tmp = train_patch[i]
                for j in range(int(count / len(train_patch[i]))):
                    shuffle(train_patch[i])
                    train_patch[i] = train_patch[i] + tmp
            shuffle(train_patch[i])
            train_patch[i] = train_patch[i][:count]
            train_labels.extend(np.full(len(train_patch[i]), i, dtype=int))
        train_patch = np.asarray(train_patch)
        train_patch = train_patch.reshape((-1, 220, PATCH_SIZE, PATCH_SIZE))

    else:
        tmp = []
        for i in range(OUTPUT_CLASSES):
            shuffle(train_patch[i])
            tmp += train_patch[i]
            train_labels.extend(np.full(len(train_patch[i]), i, dtype=int))
        train_patch = np.asarray(tmp)

    return train_patch, train_labels


def save_as_tfrecord(name, images, labels):
    """
    Converts a dataset to tfrecords.
    """
    if len(images) != len(labels):
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], len(labels)))

    file_name = name + str(PATCH_SIZE) + str(Utils.oversample) + '.tfrecords'
    print ('Writing: ' + file_name)

    with tf.python_io.TFRecordWriter(os.path.join(DATA_PATH, file_name)) as TFWriter:
        for index in range(len(images)):
            image_raw = images[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]])),
                             'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                             }))
            TFWriter.write(example.SerializeToString())


# Load data sets

DATA_PATH = Utils.data_path
input_mat = scio.loadmat(os.path.join(DATA_PATH, Utils.data_file + '.mat'))[Utils.data_name]
target_mat = scio.loadmat(os.path.join(DATA_PATH, Utils.data_file + '_gt.mat'))[Utils.data_name + '_gt']

# Define global variables

HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
PATCH_SIZE = Utils.patch_size
PATCH_IDX = int((PATCH_SIZE - 1) / 2)
COUNT = Utils.patch_class
OUTPUT_CLASSES = target_mat.max()
TEST_FRAC = Utils.test_frac
TRAIN_PATCH, TRAIN_LABELS, TEST_PATCH, TEST_LABELS = [], [], [], []
CLASSES = []
OVERSAMPLE = Utils.oversample

# Scale the input between [0,1]

input_mat = input_mat.astype(float)
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)

# Calculate the mean of each channel for normalization

MEAN_ARRAY = np.ndarray(shape=(BAND,), dtype=float)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[:, :, i])

# Collect all available patches of each class from the given image (Ignore patches of margin or unknown target)

for i in range(OUTPUT_CLASSES):
    CLASSES.append([])

for i in range(HEIGHT - PATCH_SIZE + 1):
    for j in range(WIDTH - PATCH_SIZE + 1):
        curr_inp = patch(i, j)  # Get current (BAND, PATCH_SIZE, PATCH_SIZE) patch
        curr_tar = target_mat[i + PATCH_IDX, j + PATCH_IDX]  # Target of central pixel of the patch
        if curr_tar != 0:  # Ignore patches with unknown land-cover type for the central pixel
            CLASSES[curr_tar - 1].append(curr_inp)

# Add marginal patches (mirror the images for marginal extension)

for i in list(range(PATCH_IDX)) + list(range(HEIGHT - PATCH_IDX, HEIGHT)):
    for j in list(range(PATCH_IDX)) + list(range(WIDTH - PATCH_IDX, WIDTH)):
        temp_x = patch_margin(i, j)
        temp_y = target_mat[i, j]
        if temp_y != 0:
            CLASSES[temp_y - 1].append(temp_x)

print (40 * '#' + '\n\nCollected patches of each class are: ')
print (130 * '-' + '\nClass\t|'),
for i in range(OUTPUT_CLASSES):
    print (str(i + 1) + '\t'),
print ('\n' + 130 * '-' + '\nNum\t|'),
for c in CLASSES:
    print (str(len(c)) + '\t'),
print ('\n' + 130 * '-' + '\n\n' + 40 * '#')

# Make a test split from each class

for c in range(OUTPUT_CLASSES):  # for each class
    class_population = len(CLASSES[c])
    test_split_size = int(class_population * TEST_FRAC)

    patches_of_current_class = CLASSES[c]
    shuffle(patches_of_current_class)  # Randomly shuffle patches in the class

    # Make training and test splits
    TRAIN_PATCH.append(patches_of_current_class[:-test_split_size])

    TEST_PATCH.extend(patches_of_current_class[-test_split_size:])
    TEST_LABELS.extend(np.full(test_split_size, c, dtype=int))

print ('\nTraining patches of each class are: ')
print (130 * '-' + '\nClass\t|'),
for i in range(OUTPUT_CLASSES):
    print (str(i + 1) + '\t'),
print ('\n' + 130 * '-' + '\nNum\t|'),
for c in TRAIN_PATCH:
    print (str(len(c)) + '\t'),
print ('\n' + 130 * '-' + '\n\n' + 40 * '#')

TRAIN_PATCH, TRAIN_LABELS = oversample(OVERSAMPLE, TRAIN_PATCH, TRAIN_LABELS, COUNT)  # Oversample

print ('\nTotal num of Training patches: %d\n' % len(TRAIN_PATCH))
print (40 * '#')
print ('\nTotal num of Test patches: %d\n' % len(TEST_PATCH))
print (40 * '#')

# Save the patches

# 1. Training data
file_name = 'Train_' + str(PATCH_SIZE) + str(Utils.oversample) + str(Utils.test_frac) + '.h5'
with h5py.File(os.path.join(DATA_PATH, file_name), 'w') as file:
    file.create_dataset('train_patch', data=TRAIN_PATCH, compression='gzip')
    file.create_dataset('train_labels', data=TRAIN_LABELS, compression='gzip', dtype='i8')
print ('Successfully save training data set!')

# 2. Test data
file_name = 'Test_' + str(PATCH_SIZE) + str(Utils.oversample) + str(Utils.test_frac) + '.h5'
with h5py.File(os.path.join(DATA_PATH, file_name), 'w') as file:
    file.create_dataset('test_patch', data=TEST_PATCH, compression='gzip')
    file.create_dataset('test_labels', data=TEST_LABELS, compression='gzip', dtype='i8')
print ('Successfully save test data set!')

'''
save_as_tfrecord('Train', TRAIN_PATCH, TRAIN_LABELS)
print 'Successfully save training data set!'
save_as_tfrecord('Test', TEST_PATCH, TEST_LABELS)
print 'Successfully save test data set!'
'''
