#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import sonnet as snt

from model.DAM_test import dam
from model.DNC import dnc
from loader import BAbITestBatchGenerator, BAbIData

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("embedding_size", 64, "Size of embedding.")
tf.flags.DEFINE_integer("hidden_size", 256, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_address_size", 128, "The number of memory slots.")
tf.flags.DEFINE_integer("memory_length_size", 48, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for bypass dropout")
tf.flags.DEFINE_integer("num_memory_blocks", 2, "Number of memory blocks.")

# Model selection.
tf.flags.DEFINE_boolean("dam", True, "Whether dam or not.")

# Testing options.
tf.flags.DEFINE_integer("batch_size", 200, "Batch size for training.")
tf.flags.DEFINE_string("name", "model", "Name of training model.")
tf.flags.DEFINE_integer("num", 97600, "Number of training iterations for Test.")


def run_model(input_data, sequence_length, output_size):
    """Runs model on input sequence."""

    access_config = {
        "memory_size": FLAGS.memory_address_size,
        "word_size": FLAGS.memory_length_size,
        "num_reads": FLAGS.num_read_heads,
        "num_writes": FLAGS.num_write_heads,
    }
    controller_config = {
        "hidden_size": FLAGS.hidden_size,
    }
    other_config = {
        "keep_prob": FLAGS.keep_prob,
        "num_memory_block": FLAGS.num_memory_blocks
    }

    if FLAGS.dam:
        core = dam.DAM(access_config, controller_config, other_config, output_size)
    else:
        core = dnc.DNC(access_config, controller_config, output_size)

    batch_size = tf.shape(input_data)[0]

    initial_state = core.initial_state(batch_size)
    output_sequence, _ = tf.nn.dynamic_rnn(
        cell=core,
        inputs=input_data,
        sequence_length=sequence_length,
        time_major=False,
        initial_state=initial_state)

    return output_sequence


def test():
    """Trains the DNC and periodically reports the loss."""

    test_data = BAbITestBatchGenerator()
    dataset = BAbIData(None, test_data.input_size, test_data.output_size, FLAGS.embedding_size)

    output_logits = run_model(dataset.processed_input_data, dataset.sequence_length, test_data.output_size)
    softmaxed = tf.nn.softmax(output_logits)

    saver = tf.train.Saver()

    # Train.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        saver.restore(sess, os.path.join('info', FLAGS.name, 'checkpoint', 'model.ckpt-' + str(FLAGS.num)))

        tasks_results = {}
        tasks_names = {}

        for t in os.listdir(test_data.test_data_dir):
            task_number, task_name, test_size = test_data.feed_data(t)
            tasks_names[task_number] = task_name
            counter = 0
            results = []

            test_data.feed_batch_size(FLAGS.batch_size)
            for idx in range(int(test_size / FLAGS.batch_size) + 1):
                if idx == int(test_size / FLAGS.batch_size):
                    if test_size % FLAGS.batch_size == 0:
                        break
                    test_data.feed_batch_size(test_size % FLAGS.batch_size)

                i_d, s_l, questions_indecies, target_mask, desired_answers = next(test_data)
                softmax_output = sess.run([softmaxed], feed_dict={
                    dataset.input_data: i_d,
                    dataset.sequence_length: s_l,
                })

                softmax_output = np.squeeze(softmax_output, axis=0)
                for astory, s_o, q_i, t_m, d_a in zip(i_d, softmax_output, questions_indecies, target_mask, desired_answers):
                    given_answers = np.argmax(s_o[t_m], axis=1)

                    answers_cursor = 0
                    for question_indx in q_i:
                        question_grade = []
                        targets_cursor = question_indx + 1
                        while targets_cursor < len(astory) and astory[targets_cursor] == test_data.target_code:
                            question_grade.append(given_answers[answers_cursor] == d_a[answers_cursor])
                            answers_cursor += 1
                            targets_cursor += 1
                        results.append(np.prod(question_grade))
                    counter += 1

            error_rate = 1. - np.mean(results)
            tasks_results[task_number] = error_rate
            print("\r%s ... %.3f%% Error Rate.\n" % (task_name, error_rate * 100))

        print("\n")
        print("%-27s%s" % ("Task", "Result"))
        print("-----------------------------------")
        for k in range(20):
            task_id = str(k + 1)
            task_result = "%.2f%%" % (tasks_results[task_id] * 100)
            print("%-27s%s" % (tasks_names[task_id], task_result))
        print("-----------------------------------")
        all_tasks_results = [v for _, v in tasks_results.iteritems()]
        results_mean = "%.2f%%" % (np.mean(all_tasks_results) * 100)
        failed_count = "%d" % (np.sum(np.array(all_tasks_results) > 0.05))

        print("%-27s%s" % ("Mean Err.", results_mean))
        print("%-27s%s" % ("Failed (err. > 5%)", failed_count))


def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    test()


if __name__ == "__main__":
    tf.app.run()
