"""Model util ops and modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def batch_invert_permutation(permutations):
    """Returns batched `tf.invert_permutation` for every row in `permutations`."""
    with tf.name_scope('batch_invert_permutation', values=[permutations]):
        perm = tf.cast(permutations, tf.float32)
        dim = int(perm.get_shape()[-1])
        n_block = int(perm.get_shape()[1])
        size = tf.cast(tf.shape(perm)[0], tf.float32)
        blocks = tf.cast(tf.shape(perm)[1], tf.float32)
        delta = tf.cast(tf.shape(perm)[-1], tf.float32)
        rg = tf.range(0, size * blocks * delta, delta, dtype=tf.float32)
        rg = tf.reshape(rg, [-1, n_block, 1])
        rg = tf.tile(rg, [1, 1, dim])
        perm = tf.add(perm, rg)
        flat = tf.reshape(perm, [-1])
        perm = tf.invert_permutation(tf.cast(flat, tf.int32))
        perm = tf.reshape(perm, [-1, n_block, dim])
        return tf.subtract(perm, tf.cast(rg, tf.int32))


def batch_gather(values, indices):
    """Returns batched `tf.gather` for every row in the input."""
    with tf.name_scope('batch_gather', values=[values, indices]):
        idx = tf.expand_dims(indices, -1)
        size = tf.shape(indices)[0]
        blocks = tf.shape(indices)[1]

        rg = tf.range(size, dtype=tf.int32)
        rg = tf.reshape(rg, (-1, 1, 1))
        rg = tf.tile(rg, [1, int(indices.get_shape()[1]), int(indices.get_shape()[-1])])
        rg = tf.expand_dims(rg, -1)

        bg = tf.range(blocks, dtype=tf.int32)
        bg = tf.reshape(bg, (1, -1, 1))
        bg = tf.tile(bg, [int(indices.get_shape()[0]), 1, int(indices.get_shape()[-1])])
        bg = tf.expand_dims(bg, -1)

        gidx = tf.concat([rg, bg, idx], -1)
        return tf.gather_nd(values, gidx)


def one_hot(length, index):
    """Return an nd array of given `length` filled with 0s and a 1 at `index`."""
    result = np.zeros(length)
    result[index] = 1
    return result


def reduce_prod(x, axis, name=None):
    """Efficient reduce product over axis.

    Uses tf.cumprod and tf.gather_nd as a workaround to the poor performance of calculating tf.reduce_prod's gradient on CPU.
    """
    with tf.name_scope(name, 'util_reduce_prod', values=[x]):
        cp = tf.cumprod(x, axis, reverse=True)
        size = tf.shape(cp)[0]
        idx1 = tf.range(tf.cast(size, tf.float32), dtype=tf.float32)
        idx2 = tf.zeros([size], tf.float32)
        indices = tf.stack([idx1, idx2], 1)
        return tf.gather_nd(cp, tf.cast(indices, tf.int32))


def layer_normalization(weights, dtype=tf.float32, reuse=False, name='layer_norm'):
    _eps = 1e-6

    with tf.variable_scope('{}'.format(name), reuse=reuse):
        scale = tf.get_variable('scale', shape=[weights.get_shape()[1]],
                                initializer=tf.constant_initializer(1.),
                                collections=[tf.GraphKeys.GLOBAL_VARIABLES],
                                dtype=dtype)
        beta = tf.get_variable('beta', shape=[weights.get_shape()[1]],
                               initializer=tf.constant_initializer(0.),
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES],
                               dtype=dtype)

    mean, var = tf.nn.moments(weights, axes=[1], keep_dims=True)
    norm_weights = (weights - mean) / tf.sqrt(var + _eps)

    return norm_weights * scale + beta


def clip_if_enabled(x, clip_value=20):
    if clip_value > 0:
        return tf.clip_by_value(x, -clip_value, clip_value)
    else:
        return x
