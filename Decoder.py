import numpy as np
import tensorflow as tf
import spectral
import matplotlib.pyplot as plt
import scipy.io as scio
import CNNmodels as models
import Utils
import os

DATA_PATH = Utils.data_path
input_image = scio.loadmat(os.path.join(DATA_PATH, Utils.data_file + '.mat'))[Utils.data_name]
output_image = scio.loadmat(os.path.join(DATA_PATH, Utils.data_file + '_gt.mat'))[Utils.data_name + '_gt']
model_name = os.path.join(Utils.model_path, '3D-CNN/3D-CNN-' + str(Utils.patch_size) + '.ckpt-50000')

height = output_image.shape[0]
width = output_image.shape[1]
BAND = Utils.bands
PATCH_SIZE = Utils.patch_size
batch_size = Utils.batch_size
PATCH_IDX = int((PATCH_SIZE - 1) / 2)

# Scaling Down the image to 0 - 1

input_image = input_image.astype(float)
input_image -= np.min(input_image)
input_image /= np.max(input_image)


def mirror(labels):
    patch_index = int((PATCH_SIZE - 1) / 2)
    output_mirror = np.zeros(((145 + PATCH_SIZE - 1), (145 + PATCH_SIZE - 1)), dtype='int32')
    # extend the margin of the origin image
    output_mirror[patch_index:(145+patch_index), patch_index:(145+patch_index)] = labels[:]
    output_mirror[patch_index:(145+patch_index), :patch_index] = labels[:, patch_index-1::-1]
    output_mirror[patch_index:(145+patch_index), (145+patch_index):] = labels[:, :(145-patch_index-1):-1]
    output_mirror[:patch_index, :] = output_mirror[(patch_index*2-1):(patch_index-1):-1, :]
    output_mirror[(145+patch_index):, :] = output_mirror[(145+patch_index-1):(145-1):-1, :]
    return output_mirror


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
    input_mirror = np.zeros(((height + PATCH_SIZE - 1), (width + PATCH_SIZE - 1), BAND))
    # extend the margin of the origin image
    input_mirror[PATCH_IDX:(height+PATCH_IDX), PATCH_IDX:(width+PATCH_IDX), :] = input_image[:]
    input_mirror[PATCH_IDX:(height+PATCH_IDX), :PATCH_IDX, :] = input_image[:, PATCH_IDX-1::-1, :]
    input_mirror[PATCH_IDX:(height+PATCH_IDX), (width+PATCH_IDX):, :] = input_image[:, :(width-PATCH_IDX-1):-1, :]
    input_mirror[:PATCH_IDX, :, :] = input_mirror[(PATCH_IDX*2-1):(PATCH_IDX-1):-1, :, :]
    input_mirror[(height+PATCH_IDX):, :, :] = input_mirror[(height+PATCH_IDX-1):(height-1):-1, :, :]
    transpose_array = np.transpose(input_mirror, (2, 0, 1))

    height_slice = slice(height_index, height_index + PATCH_SIZE)
    width_slice = slice(width_index, width_index + PATCH_SIZE)
    patches = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patches.shape[0]):
        mean_normalized_patch.append(patches[i] - MEAN_ARRAY[i])

    return np.array(mean_normalized_patch)


def decoder():
    with tf.Graph().as_default():
        x_placeholder = tf.placeholder(tf.float32, shape=(None, Utils.patch_size, Utils.patch_size, Utils.bands))
#        y_placeholder = tf.placeholder(tf.int32, shape=None)

        logits = models.cnn_2d(x_placeholder, False)
        softmax = tf.nn.softmax(logits)
#        eval_all = models.evaluation(logits, y_placeholder)

        saver = tf.train.Saver()

        with tf.Session() as sess:

            saver.restore(sess, model_name)

            outputs = np.zeros((height, width))
            predicted = [[0 for _ in range(width)] for _ in range(height)]

            for i in range(height):
                for j in range(width):
                    target = int(output_image[i, j])
                    if target != 0:
                        image_patch = patch_margin(i, j)
                        # print image_patch
                        prediction = sess.run(softmax, feed_dict={x_placeholder: image_patch})
                        # print prediction
                        temp = np.argmax(prediction) + 1
                        # print temp1
                        outputs[int(i + PATCH_SIZE / 2)][int(j + PATCH_SIZE / 2)] = temp
                        predicted[int(i + PATCH_SIZE / 2)][int(j + PATCH_SIZE / 2)] = prediction
                print ('Now progress: %.2f ' % (float(i) / (height - PATCH_SIZE) * 100) + '%')

    return outputs, predicted

def cr():
    """
    Collaborative Representation.
    An alternative classifier compare with Softmax
    
    Proj_M = inv(train_data'*train_data+kappa*eye(size(train_data,2)))*train_data';
    
    coef         =  Proj_M*test_data;
    for ci = 1:max(Dlabels)
        coef_c   =  coef(Dlabels==ci);
        Dc       =  D(:,Dlabels==ci);
        error(ci) = norm(y-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
    end

    index      =  find(error==min(error));
    id         =  index(1);
    
    """

    


	
	
# Calculate the mean of each channel for normalization
MEAN_ARRAY = np.ndarray(shape=(Utils.bands,), dtype=float)
for i in range(Utils.bands):
    MEAN_ARRAY[i] = np.mean(input_image[:, :, i])

# Prediction & show image
predicted_image, predicted_results = decoder()

# Save result
ground_truth = spectral.imshow(classes=output_image, figsize=(5, 5))
plt.savefig('gt.png')

ground_truth_mirror = spectral.imshow(classes=mirror(output_image), figsize=(5, 5))
plt.savefig('gt_mirror.png')

predict_image = spectral.imshow(classes=predicted_image.astype(int), figsize=(5, 5))
plt.savefig('predict.png')
