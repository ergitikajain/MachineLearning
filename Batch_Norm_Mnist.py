from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)


number_of_iteration = 50000
batch_size = 100


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages


def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 200  # fully connected layer


def deepnn(X):
    # The model
    # batch norm scaling is not useful with relus
    # batch norm offsets are used instead of biases
    pkeep_conv = tf.placeholder(tf.float32)
    iter = tf.placeholder(tf.int32)
    # test flag for batch norm
    tst = tf.placeholder(tf.bool)

    stride = 1  # output is 28x28
    num_channels = 1
    img_size = 28
    x_image = tf.reshape(X, [-1, img_size, img_size, num_channels])
    W_conv1 = weight_variable([6, 6, 1, K])  # 6x6 patch, 1 input channel, K output channels
    b_conv1 = bias_variable([K])
    h_conv1 = conv2d(x_image, W_conv1, stride)
    bn_1, update_ema1 = batchnorm(h_conv1, tst, iter, b_conv1, convolutional=True)
    relu_1 = tf.nn.relu(bn_1)
    h_dropout1 = tf.nn.dropout(relu_1, pkeep_conv, compatible_convolutional_noise_shape(relu_1))

    stride = 2  # output is 14x14
    W_conv2 = weight_variable([5, 5, K, L])
    b_conv2 = bias_variable([L])
    h_conv2 = conv2d(h_dropout1, W_conv2, stride)
    bn_2, update_ema2 = batchnorm(h_conv2, tst, iter, b_conv2, convolutional=True)
    relu_2 = tf.nn.relu(bn_2)
    h_dropout2 = tf.nn.dropout(relu_2, pkeep_conv, compatible_convolutional_noise_shape(relu_2))
    stride = 2  # output is 7x7

    W_conv3 = weight_variable([4, 4, L, M])
    b_conv3 = bias_variable([M])
    h_conv3 = conv2d(h_dropout2, W_conv3, stride)
    bn_3, update_ema3 = batchnorm(h_conv3, tst, iter, b_conv3, convolutional=True)
    relu_3 = tf.nn.relu(bn_3)
    h_dropout3 = tf.nn.dropout(relu_3, pkeep_conv, compatible_convolutional_noise_shape(relu_3))

    # reshape the output from the third convolution for the fully connected layer
    h_dropout3_flatten = tf.reshape(h_dropout3, shape=[-1, 7 * 7 * M])
    W_fc1 = weight_variable([7 * 7 * M, N])
    b_fc1 = bias_variable([N])
    h_fc1 = tf.matmul(h_dropout3_flatten, W_fc1)
    bn_4, update_ema4 = batchnorm(h_fc1, tst, iter, b_fc1)
    relu_4 = tf.nn.relu(bn_4)
    # dropout probability
    pkeep = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(relu_4, pkeep)

    W_fc2 = weight_variable([N, 10])
    b_fc2 = bias_variable([10])
    Y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = tf.nn.softmax(Y_logits)

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)
    return  y_conv, update_ema, pkeep, pkeep_conv, tst, iter


def main(_):
    # Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
    mnist = mnist_data.read_data_sets("MNIST_data", one_hot=True)

    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, shape=[None, 784])
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, update_ema, pkeep, pkeep_conv, tst, iter = deepnn(X)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100

    with tf.name_scope('adam_optimizer'):
        # training step
        # the learning rate is: # 0.0001 + 0.03 * (1/e)^(step/1000)), i.e. exponential decay from 0.03->0.0001
        lr = 0.0001 +  tf.train.exponential_decay(0.02, iter, 1600, 1/math.e)
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        # accuracy of the trained model, between 0 (worst) and 1 (best)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('/tmp/mnist_improved/train')
    train_writer.add_graph(tf.get_default_graph())
    test_writer = tf.summary.FileWriter('/tmp/mnist_improved/test')
    test_writer.add_graph(tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            batch = mnist.train.next_batch(100)
            if i % 100 == 0:

                summary, train_accuracy = sess.run([merged, accuracy], feed_dict={X: batch[0], Y_: batch[1], iter: i, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
                train_writer.add_summary(summary, i)

                summary2, test_accuracy = sess.run([merged, accuracy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
                test_writer.add_summary(summary2, i)

                print('step %d, training accuracy %g' % (i, train_accuracy))
                print('Accuracy at step %s: %s' % (i, test_accuracy))

            # the backpropagation training step
            sess.run(train_step, {X: batch[0], Y_: batch[1], tst: False, iter: i, pkeep: 0.75, pkeep_conv: 1.0})
            sess.run(update_ema, {X: batch[0], Y_: batch[1], tst: False, iter: i, pkeep: 1.0,  pkeep_conv: 1.0})


        print('test accuracy %g' % accuracy.eval(
            feed_dict={X: mnist.test.images, Y_: mnist.test.labels, tst: True, pkeep: 1.0, pkeep_conv: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

