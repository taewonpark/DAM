from __future__ import division

import re
import os
import random
import pickle

import numpy as np
import tensorflow as tf
from itertools import chain, izip_longest
from preprocess import bAbI_preprocessing


def load(path):
    return pickle.load(open(path, 'rb'))


class BAbIData(object):
    def __init__(self, batch_size, input_size, output_size, embedding_size):
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size

        self.input_data = tf.placeholder(tf.int32, [batch_size, None], name='input')
        self.target_output = tf.placeholder(tf.int32, [batch_size, None], name='targets')
        self.sequence_length = tf.placeholder(tf.int32, [batch_size], name='sequence_length')
        self.loss_mask = tf.placeholder(tf.float32, [batch_size, None])
        self.original_loss_mask = tf.placeholder(tf.float32, [batch_size, None])

        self._build_var()

        self.processed_input_data = tf.nn.embedding_lookup(self.embedding_matrix, self.input_data)
        self.one_hot_input_data = tf.one_hot(self.input_data, self.input_size, dtype=tf.float32)
        self.processed_target_data = tf.one_hot(self.target_output, self.output_size, dtype=tf.float32)

        self.eps = tf.constant(np.finfo(np.float32).eps, dtype=tf.float32)

    def _build_var(self):
        self.embedding_matrix = tf.get_variable(name='embedding_matrix',
                                                shape=[self.input_size, self.embedding_size],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())

    def cost(self, logit):
        loss = tf.reduce_mean(
            self.loss_mask * tf.nn.softmax_cross_entropy_with_logits(labels=self.processed_target_data,
                                                                     logits=logit))
        return loss

    def check_cost(self, logit):
        return tf.reduce_sum(
            self.original_loss_mask * tf.nn.softmax_cross_entropy_with_logits(labels=self.processed_target_data,
                                                                              logits=logit)
        ) / (tf.reduce_sum(self.original_loss_mask) + self.eps)


class BAbIBatchGenerator(object):
    def __init__(self, batch_size, p_re,
                 shuffle=True):
        self.batch_size = batch_size
        self.p_re = p_re
        self.shuffle = shuffle

        data_dir = 'bAbI_data'
        if not os.path.exists(data_dir):
            bAbI_preprocessing(data_dir)

        self.dataset = load(os.path.join(data_dir, 'train', 'train.pkl'))
        self.lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
        self.target_code = self.lexicon_dict['-']
        self.question_code = self.lexicon_dict['?']
        self.rest_code = self.lexicon_dict['.']

        random.shuffle(self.dataset)

        self.count = 0

        self.limit = int(len(self.dataset)/self.batch_size)

    def increase_count(self):
        self.count += 1
        if self.count >= self.limit:
            if self.shuffle:
                random.shuffle(self.dataset)
            self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.dataset[self.count * self.batch_size: (self.count + 1) * self.batch_size]

        input_vec = [sample['inputs'] for sample in samples]

        seq_len = map(len, input_vec)

        input_vec = np.array(list(izip_longest(*input_vec, fillvalue=0)), dtype=np.int32)
        input_vec = np.transpose(input_vec, (1, 0))

        target_mask = (input_vec == self.target_code)
        original_weights = target_mask.astype(np.float32)

        temp_output = [sample['outputs'] for sample in samples]
        output_vec = np.copy(input_vec)
        output_vec[target_mask] = list(chain.from_iterable(temp_output))

        if self.p_re:
            weights_vec = (input_vec != 0)
            temp_weights = np.random.choice(2, np.size(input_vec), p=[1 - self.p_re, self.p_re])
            weights_vec = weights_vec & np.reshape(temp_weights, (self.batch_size, -1))
            weights_vec = weights_vec.astype(np.float32)

            question_mask = [sample['question_mask'] for sample in samples]
            question_mask = np.array(list(izip_longest(*question_mask, fillvalue=0)), dtype=np.int32)
            question_mask = np.transpose(question_mask, (1, 0))

            weights_vec[question_mask] = 0.0
            weights_vec[input_vec == self.rest_code] = 0.0
            weights_vec[target_mask] = np.sum(weights_vec == 1, dtype=np.float32) / len(
                list(chain.from_iterable(temp_output)))
        else:
            weights_vec = original_weights

        self.increase_count()

        return (
            np.reshape(input_vec, (self.batch_size, -1)),
            np.reshape(output_vec, (self.batch_size, -1)),
            np.reshape(seq_len, (-1)),
            np.reshape(weights_vec, (self.batch_size, -1)),
            np.reshape(original_weights, (self.batch_size, -1)),
        )

    # Python 2 needs 'next()' to use iter.
    def next(self):
        return self.__next__()

    @property
    def input_size(self):
        return len(self.lexicon_dict)

    @property
    def output_size(self):
        return len(self.lexicon_dict)

    @property
    def data_size(self):
        return len(self.dataset)


class BAbITestBatchGenerator(object):
    def __init__(self):

        data_dir = 'bAbI_data'
        if not os.path.exists(data_dir):
            bAbI_preprocessing(data_dir)

        self.batch_size = 0
        self.test_data_dir = os.path.join(data_dir, 'test')
        self.dataset = None
        self.lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
        self.target_code = self.lexicon_dict['-']
        self.question_code = self.lexicon_dict['?']

        self.count = None

    def feed_data(self, task_dir):
        self.count = 0

        cur_task_dir = os.path.join(self.test_data_dir, task_dir)
        task_regexp = r'qa([0-9]{1,2})_([a-z\-]*)_test.txt.pkl'
        task_filename = os.path.basename(task_dir)
        task_match_obj = re.match(task_regexp, task_filename)
        task_number = task_match_obj.group(1)
        task_name = task_match_obj.group(2).replace('-', ' ')

        self.dataset = load(cur_task_dir)

        return task_number, task_name, len(self.dataset)

    def feed_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.dataset[self.count: self.count + self.batch_size]
        self.count += self.batch_size

        input_vec = [sample['inputs'] for sample in samples]

        seq_len = map(len, input_vec)

        input_vec = np.array(list(izip_longest(*input_vec, fillvalue=0)), dtype=np.int32)
        input_vec = np.transpose(input_vec, (1, 0))

        questions_indecies = []
        for i in input_vec:
            q = np.argwhere(i == self.question_code)
            questions_indecies.append(np.reshape(q, (-1,)))

        desired_answers = [np.array(sample['outputs']) for sample in samples]
        target_mask = (input_vec == self.target_code)

        return (
            np.reshape(input_vec, (self.batch_size, -1)),
            np.reshape(seq_len, (-1)),
            questions_indecies,
            target_mask,
            desired_answers,
        )

    # Python 2 needs 'next()' to use iter.
    def next(self):
        return self.__next__()

    @property
    def input_size(self):
        return len(self.lexicon_dict)

    @property
    def output_size(self):
        return len(self.lexicon_dict)

    @property
    def data_size(self):
        return len(self.dataset)


class AlgorithmicData(object):
    def __init__(self, batch_size, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name='input')
        self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')
        self.sequence_length = tf.placeholder(tf.int32, [batch_size], name='sequence_length')
        self.loss_mask = tf.placeholder(tf.float32, [batch_size, None])
        self.original_loss_mask = tf.placeholder(tf.float32, [batch_size, None])

        self.eps = tf.constant(np.finfo(np.float32).eps, dtype=tf.float32)

    def cost(self, logit):
        return tf.reduce_mean(
            self.loss_mask * tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_output, logits=logit), axis=-1))

    def check_cost(self, logit):
        return tf.reduce_sum(
            self.original_loss_mask * tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_output, logits=logit), axis=-1)
        ) / (tf.reduce_sum(self.original_loss_mask) * self.output_size + self.eps)


class Copy(object):
    def __init__(self, batch_size, p_re, bit_w=8, min_length=8, max_length=32, seed=0xABC):
        self.batch_size = batch_size
        self.p_re = p_re
        self.bit_w = bit_w
        self.min_length = min_length
        self.max_length = max_length + 1

        self.io_size = self.bit_w + 2

        self.seed = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):

        item_lengths = self.seed.randint(self.min_length, self.max_length, [self.batch_size])
        seq_len = (item_lengths + 1) * 2

        input_vec = []
        output_vec = []
        weights_vec = []
        original_weights = []

        for item_length, s_l in zip(item_lengths, seq_len):

            bit = self.seed.randint(0, 2, [(item_length + 1) * 2, self.io_size])

            # Set input flag, output flag part to 0
            bit[:, -2:] = 0
            # Set flag bits position to 0
            bit[0::item_length + 1, :] = 0
            # Set input flag part to 1
            bit[0, -2] = 1
            # Set output flag part to 1
            bit[item_length + 1, -1] = 1
            # Set output part to 0
            bit[item_length + 2:, :] = 0

            input_vec.append(np.copy(bit))

            bit[item_length + 2:, :] = bit[1:item_length + 1]

            output_vec.append(np.copy(bit))

            weight = np.zeros([(item_length + 1) * 2], dtype=np.float32)
            weight[item_length + 2:] = 1

            original_weights.append(weight)

            if self.p_re:
                alpha_weights = np.random.choice(2, np.size(weight), p=[1 - self.p_re, self.p_re])
                alpha_weights = alpha_weights.astype(np.float32)
                alpha_weights[s_l:] = 0
                alpha_weights[0::item_length + 1] = 0
                alpha_weights[weight == 1] = 0
                weights_vec.append(alpha_weights)

        input_vec = np.transpose(list(izip_longest(*input_vec, fillvalue=np.zeros([self.io_size]))), [1, 0, 2])
        output_vec = np.transpose(list(izip_longest(*output_vec, fillvalue=np.zeros([self.io_size]))), [1, 0, 2])
        original_weights = np.transpose(list(izip_longest(*original_weights, fillvalue=0)), [1, 0])

        if self.p_re:
            weights_vec = np.transpose(list(izip_longest(*weights_vec, fillvalue=0)), [1, 0])
            alpha = np.sum(weights_vec) / np.sum(original_weights)
            alpha = alpha if alpha >= 1 else 1.0
            weights_vec[original_weights == 1] = alpha
        else:
            weights_vec = np.copy(original_weights)

        return (
            input_vec,
            output_vec,
            seq_len,
            weights_vec,
            original_weights,
        )

    # Python 2 needs 'next()' to use iter.
    def next(self):
        return self.__next__()

    @property
    def input_size(self):
        return self.io_size

    @property
    def output_size(self):
        return self.io_size


class AssociativeRecall(object):
    def __init__(self, batch_size, p_re, bit_w=8, item_bit=3, min_length=2, max_length=8, seed=0xABC):
        self.batch_size = batch_size
        self.p_re = p_re
        self.bit_w = bit_w
        self.item_bit = item_bit
        self.min_item = min_length
        self.max_item = max_length + 1

        self.item = self.item_bit + 1
        self.io_size = self.bit_w + 2

        self.seed = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):

        num_items = self.seed.randint(self.min_item, self.max_item, [self.batch_size])
        seq_len = (num_items + 2) * self.item

        input_vec = []
        output_vec = []
        weights_vec = []
        original_weights = []

        for n_items, s_l in zip(num_items, seq_len):

            bit = self.seed.randint(0, 2, [n_items * self.item, self.io_size])

            # Set input flag, output flag part to 0
            bit[:, -2:] = 0
            # Set input flag bit to 0
            bit[0::self.item, :] = 0
            # Set input flag part to 1
            bit[0::self.item, -2] = 1

            # Select item
            seleted_item = self.seed.randint(0, n_items-1)

            # Concatenate items with selected item.
            bit = np.concatenate([bit,
                                  bit[seleted_item * self.item:(seleted_item + 1) * self.item, :],
                                  np.zeros([self.item, self.bit_w+2])], axis=0)

            bit[n_items * self.item::self.item, :] = 0
            bit[n_items * self.item: (n_items + 2) * self.item:self.item, -1] = 1

            input_vec.append(np.copy(bit))

            bit[-self.item_bit:, :-2] = \
                bit[(seleted_item + 1) * self.item + 1:(seleted_item + 2) * self.item, :-2]

            output_vec.append(np.copy(bit))

            weight = np.zeros([(n_items + 2) * self.item], dtype=np.float32)
            weight[-self.item_bit:] = 1

            original_weights.append(weight)

            if self.p_re:
                alpha_weights = np.random.choice(2, np.size(weight), p=[1 - self.p_re, self.p_re])
                alpha_weights = alpha_weights.astype(np.float32)
                alpha_weights[-(2 * self.item):] = 0
                alpha_weights[0::self.item] = 0
                alpha_weights[weight == 1] = 0
                weights_vec.append(alpha_weights)

        input_vec = np.transpose(list(izip_longest(*input_vec, fillvalue=np.zeros([self.io_size]))), [1, 0, 2])
        output_vec = np.transpose(list(izip_longest(*output_vec, fillvalue=np.zeros([self.io_size]))), [1, 0, 2])
        original_weights = np.transpose(list(izip_longest(*original_weights, fillvalue=0)), [1, 0])

        if self.p_re:
            weights_vec = np.transpose(list(izip_longest(*weights_vec, fillvalue=0)), [1, 0])
            alpha = np.sum(weights_vec) / np.sum(original_weights)
            alpha = alpha if alpha >= 1 else 1.0
            weights_vec[original_weights == 1] = alpha
        else:
            weights_vec = np.copy(original_weights)

        return (
            input_vec,
            output_vec,
            seq_len,
            weights_vec,
            original_weights,
        )

    # Python 2 needs 'next()' to use iter.
    def next(self):
        return self.__next__()

    @property
    def input_size(self):
        return self.io_size

    @property
    def output_size(self):
        return self.io_size


class RepresentationRecall(object):
    def __init__(self, batch_size, num_segment, bit_w=64, num_bits=8, min_length=8, max_length=16, seed=0xABC):

        assert num_segment % 2 == 0

        self.batch_size = batch_size
        self.num_segment = num_segment
        self.bit_w = bit_w
        self.num_bits = num_bits
        self.min_length = min_length
        self.max_length = max_length + 1

        self.subpart_bit = self.bit_w // self.num_segment
        self.max_value = 2 ** self.subpart_bit - 1

        self.subpart_candidate = [range(i * self.subpart_bit, (i + 1) * self.subpart_bit) for i in range(self.num_segment)]

        self.seed = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):

        length = self.seed.randint(self.min_length, self.max_length, [self.batch_size])
        seq_len = self.num_bits + length + 2

        input_vec = []
        output_vec = []
        original_weights = []

        s_l = max(seq_len)

        for l in length:

            # Construct Unique Bits.
            data = []
            for i in range(self.num_segment):

                datum = None
                last_size = 0
                while last_size != self.num_bits:
                    temp = self.seed.randint(1, self.max_value, size=(self.num_bits - last_size))
                    if datum is not None:
                        datum = np.concatenate((temp, datum))
                    else:
                        datum = temp

                    datum = np.unique(datum)
                    last_size = datum.size

                datum = datum.reshape(-1, 1).view(np.uint8)
                datum = np.unpackbits(np.flip(datum, axis=-1)[:, -int(np.ceil(self.subpart_bit / 8)):], axis=-1)
                datum = datum[:, -self.subpart_bit:]
                data.append(datum.astype(np.float32))

            data = np.array(data, dtype=np.float32)  # [num_segments, num_bits, subpart_bit]

            input_datum = np.zeros([s_l, self.input_size], dtype=np.float32)
            output_datum = np.zeros([s_l, self.output_size], dtype=np.float32)

            # Input Stage.
            input_datum[0, -2] = 1  # Input flag by 1.
            input_datum[1:self.num_bits + 1, :-2] = np.concatenate(data, axis=-1)

            # Query Stage.
            input_datum[self.num_bits + 1, -1] = 1
            query_index = self.seed.randint(low=1, high=self.num_bits + 1, size=[l])

            input_datum[self.num_bits + 2: self.num_bits + 2 + l, :] = input_datum[query_index]

            sequence_index = [i for i in range(self.num_bits + 2, self.num_bits + 2 + l) for _ in range(self.output_size)]

            subpart_index = [np.sort(self.seed.choice(self.num_segment, size=int(self.num_segment/2), replace=False)) for _ in range(l)]
            subpart_index = np.concatenate(subpart_index)
            subpart_index = [self.subpart_candidate[i] for i in subpart_index]
            subpart_index = np.concatenate(subpart_index)

            output_datum[self.num_bits + 2: self.num_bits + 2 + l, :] = np.reshape(input_datum[sequence_index, subpart_index], (l, -1))
            input_datum[sequence_index, subpart_index] = 0.0

            input_vec.append(input_datum)
            output_vec.append(output_datum)

            # Loss Mask.
            weight = np.zeros([s_l], dtype=np.float32)
            weight[self.num_bits + 2: self.num_bits + 2 + l] = 1
            original_weights.append(weight)

        weights_vec = original_weights

        return (
            np.array(input_vec, dtype=np.float32),
            np.array(output_vec, dtype=np.float32),
            np.array(seq_len, dtype=np.int32),
            np.array(weights_vec, dtype=np.float32),
            np.array(original_weights, dtype=np.float32)
        )

    def next(self):
        return self.__next__()

    @property
    def input_size(self):
        return self.subpart_bit * self.num_segment + 2

    @property
    def output_size(self):
        return int(self.subpart_bit * self.num_segment / 2)


if __name__ == '__main__':
    pass
