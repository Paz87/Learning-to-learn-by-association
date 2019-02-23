"""
Copied from the file architectures.py from https://github.com/haeusser/learning_by_association for learning to learn by association
with the following Copyright header:
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Definitions and utilities for the MNIST model.

This file contains functions that are needed for semisup training and evalutaion
on the fashion MNIST dataset.
They are used in train_eval.py.

"""

from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
from . import data_dirs
import scipy.io
import os
import subprocess
import h5py

DATADIR = data_dirs.fashion_mnist

NUM_LABELS = 10
IMAGE_SHAPE = [28, 28, 1]


def prepare_h5py(train_image, test_image, data_dir, shape=None):
    image = np.concatenate((train_image, test_image), axis=0).astype(np.uint8)

    print('Preprocessing data...')

    # import progressbar
    from time import sleep
    # bar = progressbar.ProgressBar(maxval=100, \
    # widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    # bar.start()

    f = h5py.File(os.path.join(data_dir, 'data.hy'), 'w')
    data_id = open(os.path.join(data_dir, 'id.txt'), 'w')
    for i in range(image.shape[0]):

        # if i%(image.shape[0]/100)==0:
        #     bar.update(i/(image.shape[0]/100))

        grp = f.create_group(str(i))
        data_id.write(str(i) + '\n')
        if shape:
            grp['image'] = np.reshape(image[i], shape, order='F')
        else:
            grp['image'] = image[i]
    # bar.finish()
    f.close()
    data_id.close()
    return


def get_data(name):
    """Utility for convenient data loading."""

    if name == 'train' or name == 'unlabeled':
        return extract_images(DATADIR +
                              '/train-images-idx3-ubyte.gz'), extract_labels(
            DATADIR + '/train-labels-idx1-ubyte.gz')
    elif name == 'test':
        return extract_images(DATADIR +
                              '/t10k-images-idx3-ubyte.gz'), extract_labels(
            DATADIR + '/t10k-labels-idx1-ubyte.gz')


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels

