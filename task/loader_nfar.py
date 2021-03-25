from __future__ import division

import numpy as np
import tensorflow as tf


class NFarthestDataL1(object):
    def __init__(self, input_size, output_size, seq_len):
        self.input_size = input_size
        self.output_size = output_size

        self.input_data = tf.placeholder(tf.float32, [None, seq_len, input_size], name='input_data')
        self.target_output = tf.placeholder(tf.int32, [None, seq_len], name='target')
        self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
        self.loss_mask = tf.placeholder(tf.float32, [None, seq_len])
        self.original_loss_mask = tf.placeholder(tf.float32, [None, seq_len])
        self.scale_factor = tf.placeholder(tf.float32, (), name="scale_factor")

        self.processed_target_data = tf.one_hot(self.target_output, self.output_size, dtype=tf.float32)
        self.processed_loss_mask = tf.expand_dims(self.loss_mask, axis=-1)

        self.eps = tf.constant(np.finfo(np.float32).eps, dtype=tf.float32)

    def show_task_loss(self, logit):
        original_loss = tf.reduce_sum(
            self.original_loss_mask * tf.nn.softmax_cross_entropy_with_logits(labels=self.processed_target_data,
                                                                              logits=logit)
        ) / (tf.reduce_sum(self.original_loss_mask) + self.eps)
        scale_loss = self.scale_factor * original_loss
        return original_loss, scale_loss

    def show_self_supervised_loss(self, logit):
        loss = tf.reduce_sum(
            self.processed_loss_mask * (self.input_data - logit)
        ) / (tf.reduce_sum(tf.cast(tf.not_equal(self.loss_mask, 0.0), dtype=tf.float32) + self.eps))
        return loss

    def task_loss(self, logit):
        loss = tf.reduce_mean(
                self.original_loss_mask * tf.nn.softmax_cross_entropy_with_logits(labels=self.processed_target_data,
                                                                                  logits=logit))
        return self.scale_factor * loss

    def self_supervised_loss(self, logit):
        loss = tf.reduce_mean(self.processed_loss_mask * (self.input_data - logit))
        return loss


class NFarthestData(object):
    def __init__(self, input_size, output_size, seq_len):
        self.input_size = input_size
        self.output_size = output_size

        self.input_data = tf.placeholder(tf.float32, [None, seq_len, input_size], name='input_data')
        self.target_output = tf.placeholder(tf.int32, [None, seq_len], name='target')
        self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
        self.loss_mask = tf.placeholder(tf.float32, [None, seq_len])
        self.original_loss_mask = tf.placeholder(tf.float32, [None, seq_len])
        self.scale_factor = tf.placeholder(tf.float32, (), name="scale_factor")

        self.processed_target_data = tf.one_hot(self.target_output, self.output_size, dtype=tf.float32)
        self.processed_loss_mask = tf.expand_dims(self.loss_mask, axis=-1)

        self.eps = tf.constant(np.finfo(np.float32).eps, dtype=tf.float32)

    def show_task_loss(self, logit):
        original_loss = tf.reduce_sum(
            self.original_loss_mask * tf.nn.softmax_cross_entropy_with_logits(labels=self.processed_target_data,
                                                                              logits=logit)
        ) / (tf.reduce_sum(self.original_loss_mask) + self.eps)
        scale_loss = self.scale_factor * original_loss
        return original_loss, scale_loss

    def show_self_supervised_loss(self, logit):
        loss = tf.reduce_sum(
            self.processed_loss_mask * tf.pow(self.input_data - logit, 2)
        ) / (tf.reduce_sum(tf.cast(tf.not_equal(self.loss_mask, 0.0), dtype=tf.float32) + self.eps))
        return loss

    def task_loss(self, logit):
        loss = tf.reduce_mean(
                self.original_loss_mask * tf.nn.softmax_cross_entropy_with_logits(labels=self.processed_target_data,
                                                                                  logits=logit))
        return self.scale_factor * loss

    def self_supervised_loss(self, logit):
        loss = tf.reduce_mean(self.processed_loss_mask * tf.pow(self.input_data - logit, 2))
        return loss


class NFarthest(object):
    def __init__(self, batch_size, batch_prob, num_vectors=8, num_dims=16, seed=None):

        self.batch_size = batch_size
        self.batch_prob = batch_prob
        self.num_vectors = num_vectors
        self.num_dims = num_dims

        self.seed = np.random.RandomState(seed) if seed is not None else np.random.RandomState(0xABC)

    def get_test_example(self, num_test_examples=3200):
        X = np.zeros((num_test_examples, self.num_vectors, self.input_size))
        y = np.zeros(num_test_examples)
        for i in range(num_test_examples):
            X_single, y_single = self.get_example()
            X[i, :] = X_single
            y[i] = y_single
        return X, y, np.array([self.num_vectors]*num_test_examples)

    @staticmethod
    def one_hot_encode(array, num_dims=8):
        one_hot = np.zeros((len(array), num_dims))
        for i in range(len(array)):
            one_hot[i, array[i]] = 1
        return one_hot

    def get_example(self):
        input_size = self.num_dims + self.num_vectors * 3
        n = self.seed.choice(self.num_vectors, 1)  # nth farthest from target vector
        labels = self.seed.choice(self.num_vectors, self.num_vectors, replace=False)
        m_index = self.seed.choice(self.num_vectors, 1)  # m comes after the m_index-th vector
        m = labels[m_index]

        # Vectors sampled from U(-1,1)
        vectors = self.seed.rand(self.num_vectors, self.num_dims) * 2 - 1
        target_vector = vectors[m_index]
        dist_from_target = np.linalg.norm(vectors - target_vector, axis=1)
        X_single = np.zeros((self.num_vectors, input_size))
        X_single[:, :self.num_dims] = vectors
        labels_onehot = self.one_hot_encode(labels, num_dims=self.num_vectors)
        X_single[:, self.num_dims:self.num_dims + self.num_vectors] = labels_onehot
        nm_onehot = np.reshape(self.one_hot_encode([n, m], num_dims=self.num_vectors), -1)
        X_single[:, self.num_dims + self.num_vectors:] = np.tile(nm_onehot, (self.num_vectors, 1))
        y_single = labels[np.argsort(dist_from_target)[-(n + 1)]]

        return X_single, y_single

    def __iter__(self):
        return self

    def __next__(self):
        X = np.zeros((self.batch_size, self.num_vectors, self.input_size), dtype=np.float32)
        y = np.zeros((self.batch_size, self.num_vectors), dtype=np.float32)
        for i in range(self.batch_size):
            X_single, y_single = self.get_example()
            X[i, :] = X_single
            y[i, -1] = y_single

        seq_len = np.array([self.num_vectors]*self.batch_size)

        original_weights = np.zeros_like(y, dtype=np.float32)
        original_weights[:, -1] = 1.0

        if self.batch_prob:
            weights_vec = self.seed.choice(2, np.size(original_weights), p=[1 - self.batch_prob, self.batch_prob])
            weights_vec = np.reshape(weights_vec, [self.batch_size, self.num_vectors])
            scale_factor = np.sum(weights_vec == 1, dtype=np.float32) / self.batch_size
            if scale_factor < 1.0:
                scale_factor = 1.0
        else:
            weights_vec = np.zeros_like(original_weights)
            scale_factor = 1.0

        return X, y, seq_len, weights_vec, original_weights, scale_factor

    # Python 2 needs 'next()' to use iter.
    def next(self):
        return self.__next__()

    @property
    def input_size(self):
        return self.num_dims + self.num_vectors * 3

    @property
    def seq_len(self):
        return self.num_vectors

    @property
    def output_size(self):
        return self.num_vectors

