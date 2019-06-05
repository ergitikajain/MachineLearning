import tensorflow as tf
import numpy as np
import scipy.ndimage

import argparse
import sys

#https://marvinler.github.io/2017/03/11/from-papers-to-github-1.html

from tensorflow.examples.tutorials.mnist import input_data


def _create_elastic_distortions(sigma, alpha):
    # Initializing with uniform distribution between -1 and 1
    x = np.random.uniform(-1, 1, size=(28, 28))
    y = np.random.uniform(-1, 1, size=(28, 28))

    # Convolving with a Gaussian filter
    x = scipy.ndimage.filters.gaussian_filter(x, sigma)
    y = scipy.ndimage.filters.gaussian_filter(y, sigma)

    # Multiplying elementwise with alpha
    x = np.multiply(x, alpha)
    y = np.multiply(y, alpha)

    return x, y

def _create_elastic_filters(n_filters, sigma=4.0, alpha=8.0):
    return [_create_elastic_distortions(sigma, alpha) for _ in range(n_filters)]


# Applies an elastic distortions filter to image
def _apply_elastic_distortions(image, filter):
    # Ensures images are of matrix representation shape
    image = np.reshape(image, (28, 28))
    res = np.zeros((28, 28))

    # filter will come out of _create_elastic_filter
    f_x, f_y = filter

    for i in range(28):
        for j in range(28):
            dx = f_x[i][j]
            dy = f_y[i][j]

            # These two variables help refactor the code
            # They are a little mind tricky; don't hesitate to take a pen and paper to visualize them
            x_offset = 1 if dx >= 0 else -1
            y_offset = 1 if dy >= 0 else -1

            # Retrieving the two closest x and y of the pixels near where the arrow ends
            y1 = j + int(dx) if 0 <= j + int(dx) < 28 else 0
            y2 = j + int(dx) + x_offset if 0 <= j + int(dx) + x_offset < 28 else 0
            x1 = i + int(dy) if 0 <= i + int(dy) < 28 else 0
            x2 = i + int(dy) + y_offset if 0 <= i + int(dy) + y_offset < 28 else 0

            # Horizontal interpolation : for both lines compute horizontal interpolation
            _interp1 = min(max(image[x1][y1] + (x_offset * (dx - int(dx))) * (image[x2][y1] - image[x1][y1]), 0), 1)
            _interp2 = min(max(image[x1][y2] + (y_offset * (dx - int(dx))) * (image[x2][y2] - image[x1][y2]), 0), 1)

            # Vertical interpolation : for both horizontal interpolations compute vertical interpolation
            interpolation = min(max(_interp1 + (dy - int(dy)) * (_interp2 - _interp1), 0), 1)

            res[i][j] = interpolation

    return res


# Creates and apply elastic distortions to the input
# images: set of images; labels: associated labels
def expand_dataset(images, labels, n_distortions):
    distortions = _create_elastic_filters(n_distortions)

    new_images_batch = np.array(
        [_apply_elastic_distortions(image, filter) for image in images for filter in distortions])
    new_labels_batch = np.array([label for label in labels for _ in distortions])

    # We don't forget to return the original images and labels (hence concatenate)
    return np.concatenate([np.reshape(images, (-1, 28, 28)), new_images_batch]), \
           np.concatenate([labels, new_labels_batch])

def get_batch(images, labels, batch_size):
    idx = np.arange(0 , len(images))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [images[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    #return images[begin: begin+batch_size], labels[begin: begin+batch_size]


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    images, labels = expand_dataset(mnist.train.images, mnist.train.labels,2)
    print(len(images))
    print(len(labels))
    return 0

if __name__ == "__main__":
    test()
