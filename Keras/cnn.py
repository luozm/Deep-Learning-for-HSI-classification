"""
CNN & FCN models for HSI classfication

@author: lzm
"""

from __future__ import print_function
import os
import numpy as np
import h5py
import Utils
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, merge, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv3D, MaxPooling3D, Conv2DTranspose
from keras.regularizers import l1, l1_l2, l2
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold


# Reading data from files
def read_file(directory, value):
    with h5py.File(directory, 'r') as readfile:
        images = readfile[value+'_patch'][:]
        labels = readfile[value+'_labels'][:]
    return images, labels


# Loading data from preprocessed files
def load_data(model_name):

    if model_name == 'fcn_2d' or model_name == 'fcn_3d':
        # load training data
        directory = os.path.join(Utils.data_path, 'Train_fcn_' + str(Utils.patch_size) + str(Utils.test_frac) + '.h5')
        train_images, train_labels = read_file(directory, 'train')
        # load test data
        directory = os.path.join(Utils.data_path, 'Test_fcn_' + str(Utils.patch_size) + str(Utils.test_frac) + '.h5')
        test_images, test_labels = read_file(directory, 'test')
        
        train_labels = np.reshape(train_labels, (train_labels.shape[0], train_labels.shape[1],  train_labels.shape[2], 1))
        test_labels = np.reshape(test_labels, (test_labels.shape[0], test_labels.shape[1],  test_labels.shape[2], 1))
        
    else:
        # load training data
        directory = os.path.join(Utils.data_path, 'Train_' + str(Utils.patch_size) + str(Utils.oversample) + str(Utils.test_frac) + '.h5')
        train_images, train_labels = read_file(directory, 'train')

        # load test data
        directory = os.path.join(Utils.data_path, 'Test_' + str(Utils.patch_size) + str(Utils.oversample) + str(Utils.test_frac) + '.h5')
        test_images, test_labels = read_file(directory, 'test')

        # Convert the shape from [num_example,channels, height, width]
        # to [num_example, height, width, channels]
        train_images = np.transpose(train_images, (0, 2, 3, 1))
        test_images = np.transpose(test_images, (0, 2, 3, 1))

        # convert class vectors to binary class matrices
        train_labels = np_utils.to_categorical(train_labels, nb_classes)
        test_labels = np_utils.to_categorical(test_labels, nb_classes)

    # Check errors
    assert train_images.shape[0] == train_labels.shape[0], (
        'Training: images.shape: %s labels.shape: %s' % (train_images.shape, train_labels.shape))
    assert test_images.shape[0] == test_labels.shape[0], (
        'Test: images.shape: %s labels.shape: %s' % (test_images.shape, test_labels.shape))
    assert train_images.shape[1:] == test_images.shape[1:], (
        'train_images.shape: %s test_images.shape: %s' % (train_images.shape[1:], test_images.shape[1:]))

    # convert input size to suit the chosen model
    if model_name == 'cnn_2d':
        input_size = train_images.shape[1:]

    elif model_name == 'cnn_3d':
        train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], train_images.shape[3], 1))
        test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], test_images.shape[3], 1))

        input_size = (train_images.shape[1], train_images.shape[2], train_images.shape[3], 1)

    elif model_name == 'fcn_2d':
        input_size = train_images.shape[1:]

    elif model_name == 'fcn_3d':
        train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], train_images.shape[3], 1))
        test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], test_images.shape[3], 1))

        input_size = (train_images.shape[1], train_images.shape[2], train_images.shape[3], 1)

    elif model_name == 'unet':
        input_size = train_images.shape[1:]

    else:
        raise ValueError('Cannot find the model, plz check your spell.')

    return train_images, train_labels, test_images, test_labels, input_size


# visualizing losses and accuracy
def visual_result(hist):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(nb_epoch)

    # Losses
    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    # use bmh, classic,ggplot for big pictures
    plt.style.available
    plt.style.use(['classic'])

    # Accuracy
    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    # use bmh, classic,ggplot for big pictures
    plt.style.available
    plt.style.use(['classic'])


# Softmax cross-entropy loss function for segmentation
def softmax_sparse_crossentropy_ignoring_first_label(y_true, y_pred):
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    y_true = tf.stack(unpacked[1:], axis=-1)

    cross_entropy = -K.sum(y_true * log, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


# Accuracy for segmentation (ignoring first label)
def sparse_accuracy(y_true, y_pred):
    classes = K.int_shape(y_pred)[-1]
    y_pred = K.reshape(y_pred, (-1, classes))

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)),
                       classes + 1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[0], tf.bool)
    y_true = tf.stack(unpacked[1:], axis=-1)

    return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))


# Define different models


# 3D-FCN model
def fcn_3d(input_shape):
    inputs = Input(input_shape)

    conv1 = Conv3D(8, kernel_size=(3, 3, 20), strides=(1, 1, 10), activation='relu')(inputs)
    pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(conv1)

    conv2 = Conv3D(32, kernel_size=(6, 6, 10), strides=(1, 1, 2), activation='relu')(pool1)
    pool2 = MaxPooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(conv2)

    conv3 = Conv3D(128, kernel_size=(3, 3, 6), activation='relu')(pool2)

    reshape = Reshape((2, 2, 128))(conv3)

    deconv1 = Conv2DTranspose(32, 3, activation='relu')(reshape)

#    up1 = UpSampling2D(size=(3, 3))(deconv1)
    deconv2 = Conv2DTranspose(8, 6, activation='relu')(deconv1)

#    up2 = UpSampling2D(size=(3, 3))(deconv2)
    deconv3 = Conv2DTranspose(nb_classes, 3)(deconv2)

    model = Model(input=inputs, output=deconv3)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss=softmax_sparse_crossentropy_ignoring_first_label,
                  optimizer=adam,
                  metrics=[sparse_accuracy])
    return model


# 2D-FCN model
def fcn_2d(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv3)

    deconv1 = Conv2DTranspose(32, 3, activation='relu', padding='valid')(conv3)
    deconv2 = Conv2DTranspose(16, 3, activation='relu', padding='valid')(deconv1)

    model = Model(input=inputs, output=deconv2)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


# U-net model
def unet(input_shape):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)

    up5 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=1)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(128, 3, activation='relu', padding='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv2], mode='concat', concat_axis=1)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv1], mode='concat', concat_axis=1)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same')(conv7)

    conv8 = Conv2D(nb_classes, 1, activation='softmax')(conv7)

    model = Model(input=inputs, output=conv8)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


# 3D-CNN model
def cnn_3d(input_shape):

    model = Sequential()
    model.add(Conv3D(16, kernel_size=(3, 3, 20), strides=(1, 1, 10), padding='valid', kernel_regularizer=l2(REG_lambda), input_shape=input_shape))
#    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=l2(REG_lambda)))
#    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 3)))

    model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=l2(REG_lambda)))
#    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=l2(REG_lambda)))
#    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 3)))

    model.add(Conv3D(64, kernel_size=(2, 2, 2), strides=(1, 1, 1), padding='same', kernel_regularizer=l2(REG_lambda)))
#    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv3D(64, kernel_size=(2, 2, 2), strides=(1, 1, 1), padding='same', kernel_regularizer=l2(REG_lambda)))
#    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
#    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


# 2D-CNN model
def cnn_2d(input_shape):

    model = Sequential()
    model.add(Conv2D(100, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(200, (3, 3), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


def inference(model_name, input_size):
    if model_name == 'cnn_2d':
        model = cnn_2d(input_size)
    elif model_name == 'cnn_3d':
        model = cnn_3d(input_size)
    elif model_name == 'unet':
        model = unet(input_size)
    elif model_name == 'fcn_2d':
        model = fcn_2d(input_size)
    elif model_name == 'fcn_3d':
        model = fcn_3d(input_size)
    else:
        raise ValueError('Cannot find the model, plz check your spell.')

    return model


# Global settings

model_type = Utils.model
nb_classes = Utils.classes
batch_size = Utils.batch_size
nb_epoch = Utils.epochs
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = 2
# convolution kernel size
kernel_size = 3
# regularization rate
REG_lambda = 0.01


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# import data from files
X_train, Y_train, X_test, Y_test, Input_shape = load_data(model_type)

# Choose a model to fit
model = inference(model_type, Input_shape)

print('model.summary:')
model.summary()

# Visualizing in TensorBoard
#tb = TensorBoard(log_dir=Utils.graph_path, histogram_freq=0, write_graph=True, write_images=False)

# Training the model
History = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                    verbose=1,
                    validation_data=(X_test, Y_test),
#                    callbacks=[tb]
                    )

# Evaluation
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

# Visualizing losses and accuracy
visual_result(History)

# Confusion Matrix
y_pred = model.predict_classes(X_test, verbose=0)
print('Classification Report:')
report = classification_report(np.argmax(Y_test, axis=-1), y_pred)
print(report)
print('Confusion Matrix:')
confusion_mtx = confusion_matrix(np.argmax(Y_test, axis=-1), y_pred)
print(confusion_mtx)


# Save model
model.save(os.path.join(Utils.model_path, str(model_type)+'-CNN-'+str(score[1])+'.h5'))
del model

# Load model
#model = load_model(os.path.join(Utils.model_path, str(model_type)+'-CNN-'+str(score[1])+'.h5'))
