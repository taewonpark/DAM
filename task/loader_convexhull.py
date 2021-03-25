from __future__ import division

import tensorflow as tf
import numpy as np
import random
import os


class ConvexHullData(object):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.input_data = tf.placeholder(tf.float32, [None, None, input_size], name='input_data')  # [batch, seq_len, size]
        self.target_output = tf.placeholder(tf.int32, [None, None], name='target')
        self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
        self.loss_mask = tf.placeholder(tf.float32, [None, None])
        self.original_loss_mask = tf.placeholder(tf.float32, [None, None])
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


class ConvexHull(object):
    def __init__(self, batch_size, p_re, max_n=20, test_n=5, curriculum=False, mode="train"):

        assert mode in ['train', 'test']

        self.batch_size = batch_size
        self.p_re = p_re
        self.max_n = max_n
        self.cur_index = -1
        self.mode = mode

        data_path = 'Convexhull_data'

        if mode == "train":
            train_dir = os.path.join(data_path, 'all_lengths_data_shuffled.txt')
            self.train_samples = self.read_file(train_dir, same_len=True)
            if curriculum:
                self.cur_index = 0
                self.train_samples.sort(key=lambda x: len(x["inputs"]))
        if mode == "test":
            assert test_n in [5, 10]
            test_dir = os.path.join(data_path, 'convex_hull_'+str(test_n)+'_test.txt')
            self.test_samples = self.read_file(test_dir)

        self.in_dim = max_n + 2 + 1  # (N_one_hot; node value; end signal)
        self.end_token = 0  # max_n + 1

    def read_file(self, filepath, same_len=False):
        all_data_blen =[]
        if same_len:
            all_data_blen = {}
        with open(filepath) as fp:
            for line in fp:
                xs = []
                ys = []
                all_items = line.strip().split()
                after_output = False
                i = 0
                while i < len(all_items):
                    if not after_output:
                        if all_items[i] == "output":
                            after_output = True
                        else:
                            xs.append([all_items[i], all_items[i+1]])
                            i += 1
                    else:
                        ys.append(all_items[i])
                    i += 1
                if len(xs) <= self.max_n:
                    if same_len:
                        if len(xs) not in all_data_blen:
                            all_data_blen[len(xs)] = []
                        all_data_blen[len(xs)].append({"inputs": np.array(xs, dtype=np.float32), "outputs": np.array(ys, dtype=np.int32)})
                    else:
                        all_data_blen.append({"inputs": np.array(xs, dtype=np.float32), "outputs": np.array(ys, dtype=np.int32)})

            return all_data_blen

    def get_train_sample_wlen(self):

        if self.cur_index < 0:
            chosen_key = random.choice(list(self.train_samples.keys()))
            samples = np.random.choice(self.train_samples[chosen_key], self.batch_size)
            data = self.prepare_sample_batch(samples)
            return data
        else:
            find = self.cur_index
            tind = self.cur_index+self.batch_size
            if tind > len(self.train_samples):
                tind = len(self.train_samples)
                find = tind-self.batch_size
                self.cur_index = 0
            else:
                self.cur_index += self.batch_size
            samples = self.train_samples[find:tind]
            data = self.prepare_sample_batch(samples)
            return data

    def get_test_sample_wlen(self):
        if self.cur_index < 0 or self.cur_index >= len(self.test_samples):
            self.cur_index = 0

        samples = self.test_samples[self.cur_index:self.cur_index+self.batch_size]
        self.cur_index += self.batch_size
        data = self.prepare_sample_batch(samples, random_mode=False)
        return data

    def prepare_sample_batch(self, samples, random_mode=True):

        batch_size = len(samples)
        input_seq_len = np.array([len(s['inputs']) for s in samples], dtype=np.int32)
        output_seq_len = np.array([len(s['outputs']) for s in samples], dtype=np.int32) + 1  # output end signal
        seq_len = input_seq_len + output_seq_len + 1  # input end signal

        max_seq_len = np.max(seq_len)

        input_vecs = np.zeros((batch_size, max_seq_len, self.in_dim), dtype=np.float32)
        output_vecs = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        original_mask = np.zeros((batch_size, max_seq_len), dtype=np.float32)

        weight_mask = np.random.choice(2, np.size(original_mask), p=[1-self.p_re, self.p_re]).astype(np.float32)
        weight_mask = weight_mask.reshape(batch_size, -1)

        for i, (s, i_s, o_s) in enumerate(zip(samples, input_seq_len, output_seq_len)):
            inputs = s['inputs']
            outputs = s['outputs']

            input_label = list(range(1, len(inputs)+1))
            label_mapping = list(range(1, len(inputs)+1))
            if random_mode:
                random.shuffle(label_mapping)
                label_mapping = dict(zip(list(range(1, len(inputs)+1)), label_mapping))
            else:
                label_mapping = dict(zip(label_mapping, label_mapping))

            input_label = [label_mapping[label] for label in input_label]
            input_label.append(0)  # end signal
            outputs = [label_mapping[label] for label in outputs]
            outputs.append(self.end_token)

            input_vecs[i, :i_s, :2] = inputs
            input_label = self.one_hot(input_label, self.max_n+1)
            input_vecs[i, :i_s+1, 2:] = input_label

            output_vecs[i, i_s+1:i_s+1+o_s] = outputs
            original_mask[i, i_s + 1:i_s + 1 + o_s] = 1.0
            weight_mask[i, i_s:] = 0.0

        if self.p_re:
            scale_factor = np.sum(weight_mask) / np.sum(original_mask)
            if scale_factor < 1.0:
                scale_factor = 1.0
        else:
            weight_mask = np.zeros_like(original_mask, dtype=np.float32)
            scale_factor = 1.0

        return (
            input_vecs,
            output_vecs,
            seq_len,
            weight_mask,
            original_mask,
            scale_factor,
        )

    @staticmethod
    def one_hot(value, dim):
        return np.eye(dim)[value]

    @property
    def input_size(self):
        return self.in_dim

    @property
    def output_size(self):
        return self.max_n + 1

    @property
    def data_size(self):
        if self.mode == 'train':
            size = 0
            for k in self.train_samples.keys():
                size += len(self.train_samples[k])
        else:
            size = len(self.test_samples)
        return size

    @property
    def test_iter(self):
        if self.mode == 'test':
            test_data_size = len(self.test_samples)
            if test_data_size % self.batch_size == 0:
                return int(test_data_size / self.batch_size)
            else:
                return int(test_data_size / self.batch_size) + 1
        else:
            return 0


if __name__ == '__main__':
    a = ConvexHull(10, 0.5)
    i_d, t_o, s_l, l_m, o_l_m, s_f = a.get_train_sample_wlen()
    # from convexhull import ConvexHull; a = ConvexHull(10, 0.1); i_d, t_o, s_l, l_m, o_l_m, s_f = a.get_train_sample_wlen()
    # from convexhull import ConvexHull; a = ConvexHull(10, 0.1, mode='test'); i_d, t_o, s_l, l_m, o_l_m, s_f = a.get_test_sample_wlen()
    print('a')
