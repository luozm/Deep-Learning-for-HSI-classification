
import numpy as np
import tensorflow as tf
import pickle as pkl
import spectral
import matplotlib.pyplot as plt
import scipy.io as scio
import CNNmodels as models
import Utils
import os


DATA_PATH = Utils.data_path
input_image = scio.loadmat(os.path.join(DATA_PATH, Utils.data_file+'.mat'))[Utils.data_name]
output_image = scio.loadmat(os.path.join(DATA_PATH, Utils.data_file+'_gt.mat'))[Utils.data_name+'_gt']
model_name = os.path.join(Utils.model_path, '2D-CNN/2D-CNN-'+str(Utils.patch_size)+'.ckpt-20000')

# input_image = np.rot90(input_image)
# output_image = np.rot90(output_image)
height = output_image.shape[0]
width = output_image.shape[1]
PATCH_SIZE = Utils.patch_size
batch_size = Utils.batch_size


# Scaling Down the image to 0 - 1

input_image = input_image.astype(float)
input_image -= np.min(input_image)
input_image /= np.max(input_image)


def mean_array(data):
    mean_arr = []
    for i in range(data.shape[2]):
        mean_arr.append(np.mean(data[:, :, i]))
    return np.array(mean_arr)


def patch(data, height_index, width_index):
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch_output = data[height_slice, width_slice, :]

    mean = mean_array(patch_output)
    mean_patch = []

    for i in range(patch_output.shape[2]):
        mean_patch.append(patch_output[:, :, i] - mean[i])

    mean_patch = np.asarray(mean_patch)
    patch_output = mean_patch.transpose((1, 2, 0))
    patch_output = patch_output.reshape(1, patch_output.shape[0], patch_output.shape[1], patch_output.shape[2])
    return patch_output


def decoder():
    with tf.Graph().as_default():
        x_placeholder = tf.placeholder(tf.float32, shape=(None, Utils.patch_size, Utils.patch_size, Utils.bands))
        logits = models.cnn_2d(x_placeholder, False)
        softmax = tf.nn.softmax(logits)

        saver = tf.train.Saver()
#        saver = tf.train.import_meta_graph(model_name + '.meta')

        with tf.Session() as sess:

            saver.restore(sess, model_name)

            outputs = np.zeros((height, width))
            predicted = [[0 for _ in range(width)]for _ in range(height)]

            for i in range(int(height-PATCH_SIZE+1)):
                for j in range(int(width-PATCH_SIZE+1)):
                    target = int(output_image[int(i+PATCH_SIZE/2), int(j+PATCH_SIZE/2)])
                    if target == 0:
                        continue
                    else:
                        image_patch = patch(input_image, i, j)
                        # print image_patch
                        prediction = sess.run(softmax, feed_dict={x_placeholder: image_patch})
                        # print prediction
                        temp = np.argmax(prediction)+1
                        # print temp1
                        outputs[int(i+PATCH_SIZE/2)][int(j+PATCH_SIZE/2)] = temp
                        predicted[int(i+PATCH_SIZE/2)][int(j+PATCH_SIZE/2)] = prediction
                print 'Now progress: %.2f ' % (float(i)/(height-PATCH_SIZE)*100)+'%'

    return outputs, predicted


# Prediction & show image
predicted_image, predicted_results = decoder()

# Save result
ground_truth = spectral.imshow(classes=output_image, figsize=(5, 5))
plt.savefig('gt.jpg')

predict_image = spectral.imshow(classes=predicted_image.astype(int), figsize=(5, 5))
plt.savefig('predict.jpg')

f_out = open('Predictions.pkl', 'ab')
pkl.dump({'11x11_aug': predicted_results}, f_out)
f_out.close()
