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
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import TensorBoard, EarlyStopping
import scipy.io as scio
import spectral


# Reading data from files
def read_file(directory, value):
    with h5py.File(directory, 'r') as readfile:
        images = readfile[value+'_patch'][:]
        labels = readfile[value+'_labels'][:]
    return images, labels


# Loading data from preprocessed files
def load_data():
    # load training data
    directory = os.path.join(Utils.data_path, 'Train_fcn_all_' + str(Utils.test_frac) + '.h5')
    images, train_labels = read_file(directory, 'train')
    # load test data
    directory = os.path.join(Utils.data_path, 'Test_fcn_all_one_' + str(Utils.test_frac) + '.h5')
    _, test_labels = read_file(directory, 'test')


#    images = np.array(images[:144, :144])
#    train_labels = np.array(train_labels[:, :144, :144])
#    test_labels = np.array(test_labels[:, :144, :144])


    train_labels = np.reshape(train_labels, (train_labels.shape[0], train_labels.shape[1],  train_labels.shape[2], 1))
    test_labels = np.reshape(test_labels, (test_labels.shape[0], test_labels.shape[1],  test_labels.shape[2], 1))

    if model_name == 'fcn_3d':
        images = np.reshape(images, (images.shape[0], images.shape[1], images.shape[2], 1))
        input_size = (images.shape[1], images.shape[2], images.shape[2], 1)
#    test_images = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
    else:
        input_size = (images.shape[0], images.shape[1], images.shape[2])

    return images, train_labels, test_labels, input_size


# visualizing losses and accuracy
def visual_result(hist):
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['sparse_accuracy']
    val_acc = hist.history['val_sparse_accuracy']
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
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1]+1)
    unpacked = tf.unstack(y_true, axis=-1)
    legal_labels = ~tf.cast(unpacked[0], tf.bool)
    y_true = tf.stack(unpacked[1:], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.sum(cross_entropy) / K.sum(tf.to_float(legal_labels))

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


def generate_batches(x, y, num_batch):
    while 1:
        np.random.shuffle(y)
        for i in range(y.shape[0] // num_batch):
            batch = slice(num_batch * i, num_batch * (i + 1))
            temp = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 1))
            temp = np.tile(temp, num_batch)
            temp = np.transpose(temp, (3, 0, 1, 2))
            yield temp, y[batch]


# Define different models

# 3D-FCN model
def fcn_3d(input_shape):
    inputs = Input(input_shape)

    conv1 = Conv3D(8, kernel_size=(3, 3, 20), strides=(1, 1, 10), activation='relu')(inputs)
    pool1 = MaxPooling3D(pool_size=(3, 3, 1), strides=(3, 3, 1))(conv1)

    conv2 = Conv3D(32, kernel_size=(6, 6, 10), strides=(1, 1, 2), activation='relu')(pool1)
    pool2 = MaxPooling3D(pool_size=(3, 3, 1), strides=(3, 3, 1))(conv2)

    conv3 = Conv3D(128, kernel_size=(3, 3, 5), strides=(1, 1, 1), activation='relu')(pool2)
    pool3 = MaxPooling3D(pool_size=(3, 3, 1), strides=(3, 3, 1))(conv3)

    reshape = Reshape((4, 4, 128))(pool3)

    up1 = UpSampling2D(size=(3, 3))(reshape)
    deconv1 = Conv2DTranspose(32, 3, activation='relu')(up1)

    up2 = UpSampling2D(size=(3, 3))(deconv1)
    deconv2 = Conv2DTranspose(8, 6, activation='relu')(up2)

    up3 = UpSampling2D(size=(3, 3))(deconv2)
    deconv3 = Conv2DTranspose(nb_classes, 3)(up3)
    deconv4 = Conv2DTranspose(nb_classes, 3)(deconv3)

    model = Model(inputs=inputs, outputs=deconv4)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss=softmax_sparse_crossentropy_ignoring_first_label,
                  optimizer=adam,
                  metrics=[sparse_accuracy])
    return model


# 2D-FCN model
def fcn_2d(input_shape):
    inputs = Input(input_shape)

    conv1 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(REG_lambda))(inputs)
#    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(act1)

    conv2 = Conv2D(32, kernel_size=(6, 6), strides=(1, 1), kernel_regularizer=l2(REG_lambda))(pool1)
#    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(act2)

    conv3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=l2(REG_lambda))(pool2)
#    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(act3)

    up1 = UpSampling2D(size=(3, 3))(pool3)
    deconv1 = Conv2DTranspose(32, 3)(up1)
    act4 = Activation('relu')(deconv1)

    up2 = UpSampling2D(size=(3, 3))(act4)
    deconv2 = Conv2DTranspose(16, 6)(up2)
    act5 = Activation('relu')(deconv2)

    up3 = UpSampling2D(size=(3, 3))(act5)
    deconv3 = Conv2DTranspose(nb_classes, 3)(up3)
    act6 = Activation('relu')(deconv3)
    deconv4 = Conv2DTranspose(nb_classes, 3)(act6)

    model = Model(inputs=inputs, outputs=deconv4)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss=softmax_sparse_crossentropy_ignoring_first_label,
                  optimizer=adam,
                  metrics=[sparse_accuracy])
    return model


# U-net model
def unet(input_shape):
    inputs = Input(input_shape)
    conv0 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(inputs)
    conv0 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv0)
    conv0 = Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu')(conv0)

    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv0)
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(conv9)

#    conv10 = Conv2D(nb_classes, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv9)

    deconv10 = Conv2DTranspose(nb_classes, kernel_size=(2, 2), strides=(1, 1), activation='relu', trainable=False)(conv9)
    conv10 = Conv2D(nb_classes, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(deconv10)
    conv10 = Conv2D(nb_classes, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv10)

    model = Model(inputs=inputs, outputs=conv10)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss=softmax_sparse_crossentropy_ignoring_first_label,
                  optimizer=adam,
                  metrics=[sparse_accuracy])
    return model


# Global settings
model_name = 'fcn_2d'
nb_classes = Utils.classes
batch_size = 2
nb_epoch = 200
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = 2
# convolution kernel size
kernel_size = 3
# regularization rate
REG_lambda = 0.02


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# import data from files
X, Y_train, Y_test, Input_shape = load_data()


# Choose a model to fit
model = unet(Input_shape)
#model = fcn_2d(Input_shape)

print('model.summary:')
model.summary()


# Visualizing in TensorBoard
tb = TensorBoard(log_dir=Utils.graph_path, histogram_freq=0, write_graph=True, write_images=False)

# Early stop
early_stop = EarlyStopping(monitor='sparse_accuracy', min_delta=1e-03, patience=4, verbose=1)

# Training the model

X_re = np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2]))
'''
History = model.fit(X_re, Y_train, batch_size=1, epochs=nb_epoch,
                    verbose=1,
                    validation_data=(X_re, Y_test),
#                    callbacks=[tb]
                    )

'''
History = model.fit_generator(
    generate_batches(X, Y_train, batch_size),
    steps_per_epoch=Y_train.shape[0]//batch_size,
    epochs=nb_epoch,
    validation_data=(X_re, Y_test),
#    callbacks=[tb, early_stop],
    verbose=1
)


# Evaluation
score = model.evaluate(X_re, Y_test, verbose=0)

print('Final Result are as follow: ')
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Visualizing losses and accuracy
#visual_result(History)

# Predictions
y_pred = model.predict(X_re)
y_pred_class = np.argmax(y_pred, axis=-1)
y_pred_class = y_pred_class.reshape(X.shape[:2])
y_pred_class += 1
y_pred_class_all = np.array(y_pred_class)

output_image = scio.loadmat(os.path.join(Utils.data_path, Utils.data_file + '_gt.mat'))[Utils.data_name + '_gt']
#output_image = np.array(output_image[:144, :144])
y_pred_class[output_image == 0] = 0

# Confusion Matrix
y_pred_2d = np.reshape(y_pred_class, -1)
y_true_2d = np.reshape(output_image, -1)
y_true_2d_test = np.reshape(Y_test, -1)
y_pred_2d_nonzero = np.array(y_pred_2d[y_true_2d_test != 0])
y_true_2d_test_nonzero = np.array(y_true_2d_test[y_true_2d_test != 0])

print('Classification Report:')
report_test = classification_report(y_true_2d_test_nonzero, y_pred_2d_nonzero)
print(report_test)

print('Confusion Matrix:')
confusion_mtx_test = confusion_matrix(y_true_2d_test_nonzero, y_pred_2d_nonzero)
print(confusion_mtx_test)

# Save result image
predict_image = spectral.imshow(classes=y_pred_class, figsize=(5, 5))
plt.savefig('predict.png')

predict_image_all = spectral.imshow(classes=y_pred_class_all, figsize=(5, 5))
plt.savefig('predict_all.png')

# Save model
model.save(os.path.join(Utils.model_path, '-FCN-ALL-'+str(score[1])+'.h5'))
#del model

# Load model
#model = load_model(os.path.join(Utils.model_path, str(model_type)+'-CNN-'+str(score[1])+'.h5'))
