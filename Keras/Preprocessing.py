# -*- coding: utf-8 -*-
"""Preprocessing data.
Load the HSI data sets and split into several patches for CNN.
@Author: lzm
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as scio
from random import shuffle
import Utils
import os
import h5py
from skimage import transform


# Define functions

def patch_margin(height_index, width_index):
    """Collect patches (including marginals)

    Returns a mean-normalized patch, the center of which
    is at (height_index, width_index)
    Inputs:
    -- height_index: row index of the center of the image patch
    -- width_index: column index of the center of the image patch
    Outputs:
    -- mean_normalized_patch: mean normalized patch of size (BAND, PATCH_SIZE, PATCH_SIZE)
    whose top left corner is at (height_index, width_index)
    """
    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patches = input_mirror_transposed[:, height_slice, width_slice]
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
        train_patch = np.array(train_patch, dtype='float32')
        train_patch = train_patch.reshape((-1, BAND, PATCH_SIZE, PATCH_SIZE))

    else:
        tmp = []
        for i in range(OUTPUT_CLASSES):
            shuffle(train_patch[i])
            tmp += train_patch[i]
            train_labels.extend(np.full(len(train_patch[i]), i, dtype=int))
        train_patch = np.array(tmp, dtype='float32')

    return train_patch, train_labels


# Load data sets

DATA_PATH = Utils.data_path
input_mat = scio.loadmat(os.path.join(DATA_PATH, Utils.data_file + '_corrected.mat'))[Utils.data_name + '_corrected']
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

input_mat = input_mat.astype('float32')
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)

if Utils.model == 'fcn_2d' or Utils.model == 'fcn_3d':

    # Randomly collect patches from the given image(Ignore all 0 label patches)
    max_sample = (HEIGHT - PATCH_SIZE + 1) * (WIDTH - PATCH_SIZE + 1)
    images = np.zeros((max_sample, PATCH_SIZE, PATCH_SIZE, BAND), dtype=float)
    labels = np.zeros((max_sample, PATCH_SIZE, PATCH_SIZE), dtype=int)
    img_idx = 0

    for h_start_px in range(HEIGHT - PATCH_SIZE + 1):
        h_end_px = h_start_px + PATCH_SIZE

        for w_start_px in range(WIDTH - PATCH_SIZE + 1):
            w_end_px = w_start_px + PATCH_SIZE
            y = target_mat[h_start_px:h_end_px, w_start_px:w_end_px]

            if y.sum() != 0:
                images[img_idx] = input_mat[h_start_px:h_end_px, w_start_px:w_end_px, :]
                labels[img_idx] = y
                img_idx += 1

    print(40 * '#')
    print('\nTotal num of none-zero label patches: %d\n' % img_idx)
    print(40 * '#')

    images = images[:img_idx]
    labels = labels[:img_idx]

    # Make a test split
    spl_idx = int(img_idx * TEST_FRAC)
    TRAIN_PATCH = images[:-spl_idx]
    TRAIN_LABELS = labels[:-spl_idx]
    TEST_PATCH = images[-spl_idx:]
    TEST_LABELS = labels[-spl_idx:]

    print('\nTotal num of Training patches: %d\n' % TRAIN_PATCH.shape[0])
    print(40 * '#')
    print('\nTotal num of Test patches: %d\n' % TEST_PATCH.shape[0])
    print(40 * '#')

    # Save the patches

    # 1. Training data
    file_name = 'Train_fcn_' + str(Utils.patch_size) + str(Utils.test_frac) + '.h5'
    print('Writing: ' + file_name)
    with h5py.File(os.path.join(DATA_PATH, file_name), 'w') as savefile:
        savefile.create_dataset('train_patch', data=TRAIN_PATCH)
        savefile.create_dataset('train_labels', data=TRAIN_LABELS, dtype='i8')
    print('Successfully save training data set!')

    # 2. Test data
    file_name = 'Test_fcn_' + str(Utils.patch_size) + str(Utils.test_frac) + '.h5'
    print('Writing: ' + file_name)
    with h5py.File(os.path.join(DATA_PATH, file_name), 'w') as savefile:
        savefile.create_dataset('test_patch', data=TEST_PATCH)
        savefile.create_dataset('test_labels', data=TEST_LABELS, dtype='i8')
    print('Successfully save test data set!')

else:
    # extend the margin of the origin image
    input_mirror = np.zeros(((HEIGHT + PATCH_SIZE - 1), (WIDTH + PATCH_SIZE - 1), BAND))
    input_mirror[PATCH_IDX:(HEIGHT + PATCH_IDX), PATCH_IDX:(WIDTH + PATCH_IDX), :] = input_mat[:]
    input_mirror[PATCH_IDX:(HEIGHT + PATCH_IDX), :PATCH_IDX, :] = input_mat[:, PATCH_IDX - 1::-1, :]
    input_mirror[PATCH_IDX:(HEIGHT + PATCH_IDX), (WIDTH + PATCH_IDX):, :] = input_mat[:, :(WIDTH - PATCH_IDX - 1):-1, :]
    input_mirror[:PATCH_IDX, :, :] = input_mirror[(PATCH_IDX * 2 - 1):(PATCH_IDX - 1):-1, :, :]
    input_mirror[(HEIGHT + PATCH_IDX):, :, :] = input_mirror[(HEIGHT + PATCH_IDX - 1):(HEIGHT - 1):-1, :, :]
    input_mirror_transposed = np.transpose(input_mirror, (2, 0, 1))
    input_mirror_transposed = input_mirror_transposed.astype('float32')

    # Calculate the mean of each channel for normalization

    MEAN_ARRAY = np.ndarray(shape=(BAND,), dtype='float32')
    for i in range(BAND):
        MEAN_ARRAY[i] = np.mean(input_mat[:, :, i])

    # Collect all available patches of each class from the given image (Ignore patches of unknown target)

    for i in range(OUTPUT_CLASSES):
        CLASSES.append([])

    # Add patches (mirror the images for marginal extension)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            temp_y = target_mat[i, j]
            if temp_y != 0:
                temp_x = patch_margin(i, j)
                CLASSES[temp_y - 1].append(temp_x)

    print (40 * '#' + '\n\nCollected patches (including marginal patches) of each class are: ')
    print (130 * '-' + '\nClass\t|', end='')
    for i in range(OUTPUT_CLASSES):
        print (str(i + 1) + '\t', end='')
    print ('\n' + 130 * '-' + '\nNum\t|', end='')
    for c in CLASSES:
        print (str(len(c)) + '\t', end='')
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

    TEST_PATCH = np.array(TEST_PATCH, dtype='float32')

    print ('\nTraining patches of each class are: ')
    print (130 * '-' + '\nClass\t|', end='')
    for i in range(OUTPUT_CLASSES):
        print (str(i + 1) + '\t', end='')
    print ('\n' + 130 * '-' + '\nNum\t|', end='')
    for c in TRAIN_PATCH:
        print (str(len(c)) + '\t', end='')
    print ('\n' + 130 * '-' + '\n\n' + 40 * '#')

    # Oversample
    TRAIN_PATCH, TRAIN_LABELS = oversample(OVERSAMPLE, TRAIN_PATCH, TRAIN_LABELS, COUNT)

    print ('\nTotal num of Training patches: %d\n' % len(TRAIN_PATCH))
    print (40 * '#')
    print ('\nTotal num of Test patches: %d\n' % len(TEST_PATCH))
    print (40 * '#')

    # Data Augmentation by rotation (0, 90, 180, 270)
    temp_patch = []
    for i in range(3):
        for j in range(len(TRAIN_PATCH)):
            temp_patch.append(transform.rotate(TRAIN_PATCH[j], (i+1)*90))

    TRAIN_PATCH = np.concatenate((TRAIN_PATCH, temp_patch))
    TRAIN_PATCH = TRAIN_PATCH.astype('float32')
    TRAIN_LABELS = np.array(TRAIN_LABELS, dtype=int)
    TRAIN_LABELS = np.tile(TRAIN_LABELS, 4)

    print ('\nTotal num of Training patches (after augmentation): %d\n' % len(TRAIN_PATCH))
    print (40 * '#')

    # Save the patches

    # 1. Training data
    file_name = 'Train_' + str(PATCH_SIZE) + str(Utils.oversample) + str(Utils.test_frac) + '.h5'
    print ('Writing: ' + file_name)
    with h5py.File(os.path.join(DATA_PATH, file_name), 'w') as file:
        file.create_dataset('train_patch', data=TRAIN_PATCH)
        file.create_dataset('train_labels', data=TRAIN_LABELS, dtype='i8')
    print ('Successfully save training data set!')

    # 2. Test data
    file_name = 'Test_' + str(PATCH_SIZE) + str(Utils.oversample) + str(Utils.test_frac) + '.h5'
    print ('Writing: ' + file_name)
    with h5py.File(os.path.join(DATA_PATH, file_name), 'w') as file:
        file.create_dataset('test_patch', data=TEST_PATCH)
        file.create_dataset('test_labels', data=TEST_LABELS, dtype='i8')
    print ('Successfully save test data set!')
