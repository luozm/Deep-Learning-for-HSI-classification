"""
Simple CNN model for HSI classfication

@author: lzm
"""

from __future__ import print_function
import numpy as np
import h5py
import Utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K


def load_data(directory, value):
    with h5py.File(directory, 'r') as readfile:
        images = readfile[value+'_patch'][:]
        labels = readfile[value+'_labels'][:]

    # Check errors
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        
    images = np.transpose(images, (0, 2, 3, 1))
    input_size = (images.shape[1], images.shape[2], images.shape[3])
    
    # convert class vectors to binary class matrices
    labels = np_utils.to_categorical(labels, nb_classes)
    
    return images, labels, input_size


def cnn_2d():

    model = Sequential()
    model.add(Conv2D(200, (3, 3),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Conv2D(100, (3, 3), padding='valid', activation='relu'))
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


nb_classes = 16

# import data from files

X_train, Y_train, input_shape = load_data('Train_'+str(Utils.patch_size)+str(Utils.oversample)+str(Utils.test_frac)+'.h5', 'train')
X_test, Y_test, input_shape2 = load_data('Test_'+str(Utils.patch_size)+str(Utils.oversample)+str(Utils.test_frac)+'.h5', 'test')

assert input_shape == input_shape2, (
        'train.shape: %s test.shape: %s' % (input_shape, input_shape2))


np.random.seed(1234)  # for reproducibility
batch_size = 100
nb_epoch = 500

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


Model = cnn_2d()
print('model.summary:')
Model.summary()

History = Model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

score = Model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

# Save model
#model.save_weights('model/2D-CNN.h5')
