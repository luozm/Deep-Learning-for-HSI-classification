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
model_name = os.path.join(Utils.model_path, '2D-CNN/2D-CNN-'+str(Utils.patch_size)+'.ckpt-50000')


height = output_image.shape[0]
width = output_image.shape[1]
PATCH_SIZE = Utils.patch_size
batch_size = Utils.batch_size


# Scaling Down the image to 0 - 1

input_image = input_image.astype(float)
input_image -= np.min(input_image)
input_image /= np.max(input_image)

    
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
    transpose_array = np.transpose(input_image, (2, 0, 1))
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patches = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patches.shape[0]):
        mean_normalized_patch.append(patches[i] - MEAN_ARRAY[i])
    patch = np.array(mean_normalized_patch)
    patch = np.transpose(patch, (1, 2, 0))
    patch = patch.reshape(1, patch.shape[0], patch.shape[1], patch.shape[2])
    return patch


def decoder():
    with tf.Graph().as_default():
        x_placeholder = tf.placeholder(tf.float32, shape=(None, Utils.patch_size, Utils.patch_size, Utils.bands))
        y_placeholder = tf.placeholder(tf.int32, shape=None)
        
        logits = models.cnn_2d(x_placeholder, False)
        softmax = tf.nn.softmax(logits)
        eval_all = models.evaluation(logits, y_placeholder)

        saver = tf.train.Saver()

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
                        image_patch = patch(i, j)
                        # print image_patch
                        prediction = sess.run(softmax, feed_dict={x_placeholder: image_patch})
                        # print prediction
                        temp = np.argmax(prediction)+1
                        # print temp1
                        outputs[int(i+PATCH_SIZE/2)][int(j+PATCH_SIZE/2)] = temp
                        predicted[int(i+PATCH_SIZE/2)][int(j+PATCH_SIZE/2)] = prediction
                print ('Now progress: %.2f ' % (float(i)/(height-PATCH_SIZE)*100)+'%')

    return outputs, predicted

    
# Calculate the mean of each channel for normalization
MEAN_ARRAY = np.ndarray(shape=(Utils.bands,), dtype=float)
for i in range(Utils.bands):
    MEAN_ARRAY[i] = np.mean(input_image[:, :, i])


# Prediction & show image
predicted_image, predicted_results = decoder()

# Save result
ground_truth = spectral.imshow(classes=output_image, figsize=(5, 5))
plt.savefig('gt.png')

predict_image = spectral.imshow(classes=predicted_image.astype(int), figsize=(5, 5))
plt.savefig('predict.png')

f_out = open('Predictions.pkl', 'ab')
pkl.dump({'11x11_aug': predicted_results}, f_out)
f_out.close()