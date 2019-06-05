########## Accuracy 99.34#######

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


filter_size1 = 3
num_filters1 = 36

filter_size2 = 3
num_filters2 = 36

filter_size3 = 3
num_filters3 = 36

num_channels = 1
img_size = 28
fc_size1 = 1024
fc_size = 1024
num_classes = 10

number_of_iteration = 50000
train_batch_size = 50


# 1.layer : convolution + max pooling
W_conv1 = weight_variable([filter_size1, filter_size1, num_channels, num_filters1])
b_conv1 = bias_variable([num_filters1])
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 2.layer : convolution + max pooling
W_conv2 = weight_variable([filter_size2, filter_size2, num_filters1, num_filters2])
b_conv2 = bias_variable([num_filters2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 3.layer : convolution + max pooling
W_conv3 = weight_variable([filter_size3, filter_size3, num_filters2, num_filters3])
b_conv3 = bias_variable([num_filters3])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


# 4. fully connected
#layer_shape = h_pool3.get_shape()
#num_features = layer_shape[1:4].num_elements()
#W_fc1 = weight_variable([num_features, fc_size1])

W_fc1 = weight_variable([4 * 4 * num_filters3, fc_size1])
b_fc1 = bias_variable([fc_size1])

h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*num_filters3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 5. fully connected
W_fc2 = weight_variable([fc_size1, num_classes])
b_fc2 = bias_variable([num_classes])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# optimization function
train_step = tf.train.AdamOptimizer(5e-4).minimize(cross_entropy)

# tensor of correct predications

# Accuracy

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/tmp/mymnist/train', sess.graph)
test_writer = tf.summary.FileWriter('/tmp/mymnist/test')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(number_of_iteration):
        batch = mnist.train.next_batch(train_batch_size)
        if i % 100 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.6})
            train_writer.add_summary(summary, i)
            print('step %d, training accuracy %g' % (i, acc))

            summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            test_writer.add_summary(summary, i)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.6})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
