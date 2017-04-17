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
from sklearn.decomposition import PCA, KernelPCA
import spectral
import matplotlib.pyplot as plt


# Define functions

# Dimensionality Reduction by PCA or KernelPCA
def dr_pca(images, is_kernel=False, kernel='linear', num_components=3):
    # Reshape 3D array to 2D
    images_2d = np.reshape(images, (-1, images.shape[-1]))

    if is_kernel:  # Using KernelPCA
        kpca = KernelPCA(n_components=num_components, kernel=kernel)
        images_pca = kpca.fit_transform(images_2d)
        # Compute variance ratio
        explained_variance = np.var(images_pca, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        print('Explained Variance Ratio: '+str(explained_variance_ratio))
        print('Sum Variance Ratio: ' + str(sum(explained_variance_ratio)))
    else:  # Using PCA
        pca = PCA(n_components=num_components)
        images_pca = pca.fit_transform(images_2d)
        explained_variance_ratio = pca.explained_variance_ratio_
        print('Explained Variance Ratio: '+str(explained_variance_ratio))
        print('Sum Variance Ratio: '+str(sum(explained_variance_ratio)))

    # convert input size to suit the chosen model
    images = np.reshape(images_pca, (images.shape[0], images.shape[1], num_components))

    return images, explained_variance_ratio


# Generate the matrix of labels
def generate_label_matrix(labels, num_sample, num_per_mtx=1):
    labels_matrix = np.zeros((num_sample, HEIGHT, WIDTH), dtype=int)
    idx, count = 0, 0
    for class_idx in range(OUTPUT_CLASSES):
        for _, sample in enumerate(labels[class_idx]):
            row, col = sample
            labels_matrix[idx, row, col] = class_idx + 1
            count += 1
            if count % num_per_mtx == 0:
                idx += 1
    labels_matrix = np.array(labels_matrix, dtype='uint8')
    return labels_matrix


# Generate the matrix of labels (all in one matrix)
def generate_label_matrix_one(labels):
    labels_matrix = np.zeros((1, HEIGHT, WIDTH), dtype=int)
    for class_idx in range(OUTPUT_CLASSES):
        for _, sample in enumerate(labels[class_idx]):
            row, col = sample
            labels_matrix[0, row, col] = class_idx + 1
    labels_matrix = np.array(labels_matrix, dtype='uint8')
    return labels_matrix


# Make a test split
def split(labels, num_classes, test_frac):
    train_y, test_y = [], []

    for class_idx in range(num_classes):  # for each class
        class_population = len(labels[class_idx])
        test_split_size = int(class_population * test_frac)
        patches_of_current_class = labels[class_idx]

        # Randomly shuffle patches in the class
        shuffle(patches_of_current_class)

        # Make training and test splits
        train_y.append(patches_of_current_class[:-test_split_size])
        test_y.append(patches_of_current_class[-test_split_size:])

    return train_y, test_y


# Save the dataset to files
def save_file(images, labels, file_name, variable_name):
    file_name = file_name + str(Utils.test_frac) + '.h5'
    print('Writing: ' + file_name)
    with h5py.File(os.path.join(DATA_PATH, file_name), 'w') as savefile:
        savefile.create_dataset(variable_name + '_patch', data=images, dtype='float32')
        savefile.create_dataset(variable_name + '_labels', data=labels, dtype='uint8')
    print('Successfully save ' + variable_name + ' data set!')


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
PCA_components = 10
CLASSES = []
num_samples, num_train_samples = 0, 0

# Scale the input between [0,1]

input_mat = input_mat.astype('float32')
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)


# Dimensionality Reduction
input_mat_pca, Variance_ratio = dr_pca(input_mat, False, num_components=PCA_components)


# Collect labels from the given image(Ignore 0 label patches)

for i in range(OUTPUT_CLASSES):
    CLASSES.append([])

# Collect labels (Ignore 0 labels)
for i in range(HEIGHT):
    for j in range(WIDTH):
        temp_y = target_mat[i, j]
        if temp_y != 0:
            CLASSES[temp_y - 1].append([i, j])

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
train_labels, test_labels = split(CLASSES, OUTPUT_CLASSES, TEST_FRAC)

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
TRAIN_LABELS = generate_label_matrix(train_labels, num_train_samples//100+1, 100)
TRAIN_LABELS_ONE = generate_label_matrix_one(train_labels)
TEST_LABELS_ONE = generate_label_matrix_one(test_labels)

print('\nTotal num of Training labels: %d\n' % num_train_samples)
print(40 * '#')
print('\nTotal num of Test labels: %d\n' % (num_samples - num_train_samples))
print(40 * '#')

# Save ground truth & image after PCA
input_image = spectral.imshow(input_mat, figsize=(5, 5))
plt.savefig('image.png')

ground_truth = spectral.imshow(classes=target_mat, figsize=(5, 5))
plt.savefig('gt.png')

pca_image = spectral.imshow(input_mat_pca, figsize=(5, 5))
plt.savefig('pca_' + str(PCA_components) + '.png')


# Save the patches
save_file(input_mat_pca, TRAIN_LABELS, 'Train_fcn_all_', 'train')
save_file(input_mat_pca, TRAIN_LABELS_ONE, 'Train_fcn_all_one_', 'train')
save_file(input_mat_pca, TEST_LABELS_ONE, 'Test_fcn_all_one_', 'test')
