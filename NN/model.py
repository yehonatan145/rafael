import tensorflow as tf
import math
import numpy as np
import collections

log_dir = 'log/'
num_time_samples = 30
num_windows = 20
num_sample_features = 6
num_labels = 25
batch_size = 100
max_steps = 100000

conv1_num_filters = 10
conv2_num_filters = 3
conv2_one_layer_length = 90
first_hid = 30
num_encode_features = 10
num_class_feature = 7


def inference(routes, labels):
    class_loss = 0.0
    kernel = tf.Variable(np.random.rand(3, 1, 6, conv1_num_filters))
    W1 = tf.Variable(np.random.rand(conv2_one_layer_length, first_hid), dtype=tf.float32)
    b1 = tf.Variable(np.zeros((1, first_hid)), dtype=tf.float32)
    W2 = tf.Variable(np.random.rand(first_hid, num_encode_features), dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1, num_encode_features)), dtype=tf.float32)
    W3 = tf.Variable(np.random.rand(num_class_feature, num_labels), dtype=tf.float32)
    b3 = tf.Variable(np.zeros((1, num_labels)), dtype=tf.float32)
    # for i in range(batch_size):
    #     if i > 0:
    #         tf.get_variable_scope().reuse_variables()
    # with tf.name_scope('conv1'):
    conv = tf.nn.conv2d(routes, tf.cast(kernel, dtype=tf.float32), [1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.relu(conv)
    # norm1 = tf.nn.lrn(conv1, 3, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # with tf.name_scope('conv2'):
    #     kernel = tf.Variable(np.random.rand(3, 20, 15))
    #     conv = tf.nn.conv1d(conv1, tf.cast(kernel, dtype=tf.float32), 1, padding='VALID')
    # biases = tf.Variable('biases', [64], tf.constant_initializer(0.1))
    # pre_activation = tf.nn.bias_add(conv, biases)
    # conv2 = tf.nn.relu(conv)
    conv2_one_layer = tf.reshape(conv1, (batch_size, num_windows, conv2_one_layer_length))
    # pooled = tf.nn.max_pool(conv2_one_layer, [1, 10, 1, 1], [1, 1, num_encode_features, 1], 'VALID')
    # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    # print(pooled)
    # with tf.name_scope('hidden1'):

    x = tf.nn.relu([tf.matmul(conv2_one_layer[k], tf.reshape(W1, (conv2_one_layer_length, first_hid)))
                    + tf.reshape(b1, (1, first_hid)) for k in range(batch_size)])

    # with tf.name_scope('hidden2'):
    feat = [tf.matmul(x[k], W2) + b2 for k in range(batch_size)]
    # print(feat)
    # features_series.append(feat)
    # with tf.name_scope('classification'):
    # print(feat)
    feat_to_pool = tf.reshape(feat, (batch_size, num_windows, num_encode_features, 1))
    feat_pool = tf.reshape(tf.nn.max_pool(feat_to_pool, [1, 10, 1, 1], [1, 20, 1, 1], 'VALID'),
                           (batch_size, num_encode_features))
    # print(feat_to_pool)
    # print(feat_pool)
    class_feats = feat_pool[:, 0:num_class_feature]
    logits = [tf.matmul(tf.reshape(class_feats[i], (1, num_class_feature)), W3) + b3 for i in range(batch_size)]
    logits = tf.reshape(logits, (batch_size, num_labels))
    # print(labels)
    # print(labels[i])
    loss = tf.losses.sparse_softmax_cross_entropy(labels, tf.reshape(logits, (batch_size, 1, 25)))
    class_loss += loss
    # logits_series.append(logits)
    # with tf.name_scope('decoder_hid'):
    #     W = tf.Variable(np.random.rand(num_encode_features, first_hid), dtype=tf.float32)
    #     b = tf.Variable(np.zeros((1, first_hid)), dtype=tf.float32)
    #     x = tf.nn.relu(tf.matmul(feat, W) + b)
    # with tf.name_scope('decoder_final'):
    #     W = tf.Variable(np.random.rand(first_hid, conv2_one_layer_length), dtype=tf.float32)
    #     b = tf.Variable(np.zeros((1, conv2_one_layer_length)), dtype=tf.float32)
    #     decoded_image = tf.nn.relu(tf.matmul(x, W) + b)
    # with tf.name_scope('decoder_loss'):
    #
    #     pass
    # loss = tf.losses.mean_squared_error(tf.reshape(images[i], (1, 784)), tf.reshape(decoded_image, (1, 784)))
    # decode_loss += loss
    return feat_pool, logits, loss, 0


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
    guessed_labels = tf.reshape(tf.cast(tf.argmax(tf.nn.softmax(logits), 1), tf.int32), (batch_size, 1))
    print(logits)
    print(labels)
    print(guessed_labels)
    # one_hot_labels = tf.reshape(tf.cast(tf.one_hot(labels, num_labels), tf.int32), (batch_size, 1, num_labels))
    # print("check shapes: \n", guessed_labels)
    # print(labels)
    return tf.reduce_sum(tf.cast(tf.equal(guessed_labels, labels), tf.int32))
    # return tf.reduce_sum(
    #     tf.cast(tf.logical_and(tf.cast(guessed_labels, tf.bool), tf.cast(one_hot_labels, tf.bool)), tf.int32))
