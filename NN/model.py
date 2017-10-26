import tensorflow as tf
import math
import numpy as np
import collections

log_dir = 'log/'
num_time_samples = 30
num_sample_features = 6
num_labels = 25
batch_size = 1000
learning_rate = 0.1
max_steps = 100000

conv1_num_filters = 6
conv2_num_filters = 3
conv2_one_layer_length = 90
first_hid = 30
num_encode_features = 10
num_class_feature = 7


def inference(routes, labels):
    features_series = []
    logits_series = []
    class_loss = 0.0
    decode_loss = 0.0
    for i in range(batch_size):
        if i > 0:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('conv1'):
            kernel = tf.Variable(np.random.rand(3, 6, 20))
            conv = tf.nn.conv1d([routes[i][0]], tf.cast(kernel, dtype=tf.float32), 1, padding='VALID')
            conv1 = tf.nn.relu(conv)

        # norm1 = tf.nn.lrn(conv1, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        with tf.name_scope('conv2'):
            kernel = tf.Variable(np.random.rand(3, 20, 15))
            conv = tf.nn.conv1d(conv1, tf.cast(kernel, dtype=tf.float32), 1, padding='VALID')
            # biases = tf.Variable('biases', [64], tf.constant_initializer(0.1))
            # pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(conv)
            conv2_one_layer = tf.reshape(conv2, (1, conv2_one_layer_length))

        # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        with tf.name_scope('hidden1'):
            W = tf.Variable(np.random.rand(conv2_one_layer_length, first_hid), dtype=tf.float32)
            b = tf.Variable(np.zeros((1, first_hid)), dtype=tf.float32)
            x = tf.nn.relu(tf.matmul(tf.reshape(conv2_one_layer, (1, conv2_one_layer_length)), W) + b)
        with tf.name_scope('hidden2'):
            W = tf.Variable(np.random.rand(first_hid, num_encode_features), dtype=tf.float32)
            b = tf.Variable(np.zeros((1, num_encode_features)), dtype=tf.float32)
            feat = tf.matmul(x, W) + b
            features_series.append(feat)
        with tf.name_scope('classification'):
            x = feat[:, 0:5]
            W = tf.Variable(np.random.rand(5, num_labels), dtype=tf.float32)
            b = tf.Variable(np.zeros((1, num_labels)), dtype=tf.float32)
            logits = tf.matmul(x, W) + b
            loss = tf.losses.sparse_softmax_cross_entropy(labels[i], tf.reshape(logits, (1, 25)))
            class_loss += loss
            logits_series.append(logits)
        with tf.name_scope('decoder_hid'):
            W = tf.Variable(np.random.rand(num_encode_features, first_hid), dtype=tf.float32)
            b = tf.Variable(np.zeros((1, first_hid)), dtype=tf.float32)
            x = tf.nn.relu(tf.matmul(feat, W) + b)
        with tf.name_scope('decoder_final'):
            W = tf.Variable(np.random.rand(first_hid, conv2_one_layer_length), dtype=tf.float32)
            b = tf.Variable(np.zeros((1, conv2_one_layer_length)), dtype=tf.float32)
            decoded_image = tf.nn.relu(tf.matmul(x, W) + b)
        with tf.name_scope('decoder_loss'):
            pass
            # loss = tf.losses.mean_squared_error(tf.reshape(images[i], (1, 784)), tf.reshape(decoded_image, (1, 784)))
            # decode_loss += loss
    return tf.stack(features_series), tf.stack(logits_series), class_loss / batch_size / 10, decode_loss / batch_size


def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    # correct =  tf.gather(labels, tf.nn.top_k(logits, 1))
    guessed_labels = tf.cast(tf.round(tf.nn.softmax(logits)), tf.int32)
    # print("check shapes: \n", guessed_digits)
    # print(labels)
    return tf.reduce_sum(
        tf.cast(tf.equal(tf.cast(guessed_labels, tf.int32),
                         tf.reshape(tf.cast(tf.one_hot(labels, num_labels), tf.int32), (batch_size, 1, num_labels))),
                tf.int32))
