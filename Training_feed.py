"""Training CNN model

Trains and Evaluates CNNs using a feed dictionary

@Author: lzm
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from six.moves import xrange  # redefined-builtin
import tensorflow as tf
import os
import CNNmodels as models
import Utils


# Define functions

def feed_dict(is_train, data_set):
    """
    Fill a feed dictionary with the actual set of images and labels
    for this particular training step.

    Inputs:
    -- is_train: bool value, whether train or not
    -- data_set: whether Train set or Test set

    Return: feed_dict
    """
    feed_images, feed_labels = data_set.next_batch(Utils.batch_size)
    return {x_placeholder: feed_images, y_placeholder: feed_labels, is_training: is_train}


def eval_full_epoch(data_set):
    """
    Runs one evaluation against the full epoch of data.

    Inputs:
    -- data_set: The set of images and labels to evaluate, from
                input_data.read_data_sets().
    """
    true_count = 0  # Counts the number of correct predictions.
    data_set.start_new_epoch()  # Start new epoch for evaluation
    steps_per_epoch = data_set.num_examples // Utils.batch_size
    num_examples = steps_per_epoch * Utils.batch_size

    for _ in xrange(steps_per_epoch):
        true_count += sess.run(EVAL, feed_dict=feed_dict(False, data_set))
    accuracy = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Accuracy: %0.04f' % (num_examples, true_count, accuracy))


# Import training & test datasets from files

# training set
data_sets = Utils.read_data_sets(os.path.join(Utils.data_path, 'Train_'+str(Utils.patch_size)+str(Utils.test_frac)+str(Utils.oversample)+'.h5'), 'train')
Training_data = data_sets

# test set
data_sets = Utils.read_data_sets(os.path.join(Utils.data_path, 'Test_'+str(Utils.patch_size)+str(Utils.test_frac)+str(Utils.oversample)+'.h5'), 'test')
Test_data = data_sets


# Built the model


# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():

    # Generate placeholders for the images and labels.

    with tf.name_scope('Inputs'):
        x_placeholder = tf.placeholder(
            tf.float32,
            shape=(Utils.batch_size, Utils.patch_size, Utils.patch_size, Utils.bands),
            name='x-input'
        )
        y_placeholder = tf.placeholder(tf.int32, shape=Utils.batch_size, name='y-input')
        is_training = tf.placeholder(tf.bool, name='is-training')  # used for dropout

    # Build a Graph that computes predictions from the inference model.
    LOGITS = models.cnn_2d(x_placeholder, is_training)
#    LOGITS = models.cnn_3d(x_placeholder, is_training)

    # Add to the Graph the Ops for loss calculation.
    LOSS = models.loss(LOGITS, y_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    TRAIN = models.training(LOSS, Utils.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    EVAL = models.evaluation(LOGITS, y_placeholder)

    # Initialize
    with tf.name_scope('Initialize'):
        INIT = tf.global_variables_initializer()

    # Build the summary operation based on the TF collection of Summaries.
    MERGE = tf.summary.merge_all()

    # Create a session for running Ops on the Graph.
    with tf.Session() as sess:

        # Run the Op to initialize the variables.
        sess.run(INIT)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        train_summary_writer = tf.summary.FileWriter(
            os.path.join(Utils.graph_path, 'train'), sess.graph, flush_secs=60)
        test_summary_writer = tf.summary.FileWriter(
            os.path.join(Utils.graph_path, 'test'), sess.graph, flush_secs=60)

        # Start the training loop.
        for step in xrange(Utils.max_steps):
            start_time = time.time()

            # Write the summaries, train, and print an overview
            if step % 50 == 0 or (step+1) == Utils.max_steps:
#                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#                run_metadata = tf.RunMetadata()
                _, loss_value, summary_str = sess.run(
                    [TRAIN, LOSS, MERGE],
                    feed_dict=feed_dict(True, Training_data),
#                    options=run_options,
#                    run_metadata=run_metadata
                )
                # Update the events file.
#                train_summary_writer.add_run_metadata(run_metadata, 'step%d' % (step+1))
                train_summary_writer.add_summary(summary_str, (step+1))

                duration = time.time() - start_time  # Time in the epoch

                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % ((step+1), loss_value, duration))

            # Run one step of the model.
            else:
                sess.run(TRAIN, feed_dict=feed_dict(True, Training_data))

            # Evaluate model (mini-batch)
            if (step+1) % 100 == 0:
                _, summary_str = sess.run([EVAL, MERGE], feed_dict=feed_dict(False, Test_data))
                test_summary_writer.add_summary(summary_str, (step+1))

            # Save a checkpoint and evaluate the model periodically.
            if (step+1) % 5000 == 0 or (step+1) == Utils.max_steps:
                saver.save(sess, os.path.join(
                    Utils.model_path,
                    '2D-CNN/2D-CNN-'+str(Utils.patch_size)+'.ckpt'),
                    global_step=step+1)

                # Evaluate against the training & test sets.
                print('Training Data Eval:')
                eval_full_epoch(Training_data)
                print('Test Data Eval:')
                eval_full_epoch(Test_data)

        train_summary_writer.close()
        test_summary_writer.close()
