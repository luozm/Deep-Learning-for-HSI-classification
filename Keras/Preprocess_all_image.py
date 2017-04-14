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


# Define functions

# Generate the matrix of labels
def generate_label_matrix(labels, num):
    labels_matrix = np.zeros((num, HEIGHT, WIDTH), dtype=int)
    idx = 0
    for class_idx in range(OUTPUT_CLASSES):
        for _, sample in enumerate(labels[class_idx]):
            row, col = sample
            labels_matrix[idx, row, col] = class_idx + 1
            idx += 1
    return labels_matrix


# Make a test split
def split(labels, num_classes):
    train_labels, test_labels = [], []

    for class_idx in range(num_classes):  # for each class
        class_population = len(labels[class_idx])
        test_split_size = int(class_population * TEST_FRAC)
        patches_of_current_class = labels[class_idx]

        # Randomly shuffle patches in the class
        shuffle(patches_of_current_class)

        # Make training and test splits
        train_labels.append(patches_of_current_class[:-test_split_size])
        test_labels.append(patches_of_current_class[-test_split_size:])

    return train_labels, test_labels


# Load data sets

DATA_PATH = Utils.data_path
input_mat = scio.loadmat(os.path.join(DATA_PATH, Utils.data_file + '_corrected.mat'))[Utils.data_name + '_corrected']
target_mat = scio.loadmat(os.path.join(DATA_PATH, Utils.data_file + '_gt.mat'))[Utils.data_name + '_gt']

# Define global variables

HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
PATCH_SIZE = Utils.patch_size
OUTPUT_CLASSES = int(target_mat.max())
TEST_FRAC = Utils.test_frac
CLASSES = []

# Scale the input between [0,1]

input_mat = input_mat.astype('float32')
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)


# Collect labels from the given image(Ignore 0 label patches)

for i in range(OUTPUT_CLASSES):
    CLASSES.append([])

# Collect labels (Ignore 0 labels)
for i in range(HEIGHT):
    for j in range(WIDTH):
        temp_y = target_mat[i, j]
        if temp_y != 0:
            CLASSES[temp_y - 1].append([i, j])

num_samples = 0
print(40 * '#' + '\n\nCollected labels of each class are: ')
print(130 * '-' + '\nClass\t|', end='')
for i in range(OUTPUT_CLASSES):
    print(str(i + 1) + '\t', end='')
print('\n' + 130 * '-' + '\nNum\t|', end='')
for c in CLASSES:
    print(str(len(c)) + '\t', end='')
    num_samples += len(c)
print('\n' + 130 * '-' + '\n\n' + 40 * '#')


# Make a test split
train_labels, test_labels = split(CLASSES, OUTPUT_CLASSES)

num_train_samples = 0
print ('\nTraining labels of each class are: ')
print (130 * '-' + '\nClass\t|', end='')
for i in range(OUTPUT_CLASSES):
    print (str(i + 1) + '\t', end='')
print ('\n' + 130 * '-' + '\nNum\t|', end='')
for c in train_labels:
    print (str(len(c)) + '\t', end='')
    num_train_samples += len(c)
print ('\n' + 130 * '-' + '\n\n' + 40 * '#')


# Generate the matrix of labels
TRAIN_LABELS = generate_label_matrix(train_labels, num_train_samples)
TEST_LABELS = generate_label_matrix(test_labels, num_samples - num_train_samples)

print('\nTotal num of Training labels: %d\n' % TRAIN_LABELS.shape[0])
print(40 * '#')
print('\nTotal num of Test labels: %d\n' % TEST_LABELS.shape[0])
print(40 * '#')

# Save the patches

# 1. Training data
file_name = 'Train_fcn_all_' + str(Utils.test_frac) + '.h5'
print('Writing: ' + file_name)
with h5py.File(os.path.join(DATA_PATH, file_name), 'w') as savefile:
    savefile.create_dataset('train_patch', data=input_mat)
    savefile.create_dataset('train_labels', data=TRAIN_LABELS, dtype='i8')
print('Successfully save training data set!')

# 2. Test data
file_name = 'Test_fcn_all_' + str(Utils.test_frac) + '.h5'
print('Writing: ' + file_name)
with h5py.File(os.path.join(DATA_PATH, file_name), 'w') as savefile:
    savefile.create_dataset('test_patch', data=input_mat)
    savefile.create_dataset('test_labels', data=TEST_LABELS, dtype='i8')
print('Successfully save test data set!')
