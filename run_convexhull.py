#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sonnet as snt
import os
import time

from model.DAM_test import dam
from model.DNC import dnc
from task.loader_convexhull import ConvexHull, ConvexHullData

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 256, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_address_size", 20, "The number of memory slots.")
tf.flags.DEFINE_integer("memory_length_size", 64, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_float("keep_prob", 0.9, "Keep probability for bypass dropout")
tf.flags.DEFINE_integer("num_memory_blocks", 6, "Number of memory blocks.")

# Optimizer parameters.
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("momentum", 0.9, "Optimizer momentum.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10, "Epsilon used for RMSProp optimizer.")
tf.flags.DEFINE_integer("clip_value", 20, "Maximum absolute value of controller and dnc outputs.")

# Model selection.
tf.flags.DEFINE_boolean("dam", True, "Whether dam or not.")

# MRL parameters.
tf.flags.DEFINE_float("p_re", 0.3, "Memory Refreshing Probability.")

# Training options.
tf.flags.DEFINE_integer("batch_size", 128, "Batch size for training.")
tf.flags.DEFINE_integer("epoch", 1, "Number of epochs to train for.")
tf.flags.DEFINE_integer("summarize", 25, "Interval for summarize.")
tf.flags.DEFINE_integer("training_iteration", 100000, "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 1000, "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_integer("checkpoint_interval", 10000, "Checkpointing step interval.")
tf.flags.DEFINE_string("name", "convexhull", "Name of training model.")


def run_model(input_data, sequence_length, input_size, target_size):
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
        "num_memory_block": FLAGS.num_memory_blocks,
        "act_fn_list": ['relu', 'relu'],
        "layer_size_list": [256, 256]
    }
    clip_value = FLAGS.clip_value

    output_size = input_size + target_size

    if FLAGS.dam:
        dnc_core = dam.DAM(access_config, controller_config, other_config, output_size, clip_value)
    else:
        dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)

    print("DNC Core: " + str(dnc_core))

    batch_size = tf.shape(input_data)[0]
    initial_state = dnc_core.initial_state(batch_size)
    output_sequence, _ = tf.nn.dynamic_rnn(
        cell=dnc_core,
        inputs=input_data,
        sequence_length=sequence_length,
        time_major=False,
        initial_state=initial_state)

    self_supervised_output = output_sequence[:, :, :input_size]
    target_output = output_sequence[:, :, input_size:]

    return self_supervised_output, target_output


def train(epoch, report_interval):
    """Trains the DNC and periodically reports the loss."""

    data = ConvexHull(FLAGS.batch_size, FLAGS.p_re, mode='train')
    test_5_data = ConvexHull(500, 0.0, test_n=5, mode='test')
    test_10_data = ConvexHull(500, 0.0, test_n=10, mode='test')
    dataset = ConvexHullData(data.input_size, data.output_size)
    self_supervised_output, target_output = run_model(dataset.input_data, dataset.sequence_length, dataset.input_size, dataset.output_size)

    print("Association Reinforcing Probability: " + str(FLAGS.p_re))
    print("Train Data Generator: " + str(data))
    print("Dataset: " + str(dataset))
    print("Number of training paramter: {0}".format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

    task_loss = dataset.task_loss(target_output)
    self_supervised_loss = dataset.self_supervised_loss(self_supervised_output)
    train_loss = task_loss + self_supervised_loss

    showed_original_loss, showed_scale_loss = dataset.show_task_loss(target_output)
    showed_self_supervised_loss = dataset.self_supervised_loss(self_supervised_output)
    tf.summary.scalar("original_loss", showed_original_loss)
    tf.summary.scalar("scale_loss", showed_scale_loss)
    tf.summary.scalar("self_supervised_loss", showed_self_supervised_loss)
    summary_op = tf.summary.merge_all()
    no_summary = tf.no_op()

    result = tf.nn.softmax(target_output, axis=-1)
    final_result = tf.argmax(result, axis=-1)

    # Set up optimizer with global norm clipping.
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    optimizer = tf.train.RMSPropOptimizer(
        FLAGS.learning_rate, momentum=FLAGS.momentum, epsilon=FLAGS.optimizer_epsilon)
    train_op = optimizer.apply_gradients(
        zip(grads, trainable_variables), global_step=global_step)

    saver = tf.train.Saver()
    save_steps = FLAGS.checkpoint_interval

    if FLAGS.checkpoint_interval > 0:
        hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=os.path.join('info', FLAGS.name, "checkpoint"),
                save_steps=save_steps,
                saver=saver)
        ]
    else:
        hooks = []

    # Train.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.train.SingularMonitoredSession(
            hooks=hooks, checkpoint_dir=os.path.join('info', FLAGS.name, "checkpoint"), config=config) as sess:

        fetches = {
            'train_op': train_op,
            'loss': showed_original_loss,
            # 'summary_op': summary_op,
            'step': global_step,
        }

        summary_writer = tf.summary.FileWriter(os.path.join('info', FLAGS.name, "log"), graph=tf.get_default_graph())

        iteration_per_epoch = int(FLAGS.training_iteration / epoch)

        # Accuracy for N=5.
        accuracy_for_5 = 0
        for _ in range(test_5_data.test_iter):
            i_d, t_o, s_l, _, o_l_m, _ = test_5_data.get_test_sample_wlen()
            prediction = sess.run(final_result, feed_dict={
                dataset.input_data: i_d,
                dataset.sequence_length: s_l
            })
            for p_, t_, o_ in zip(prediction, t_o, o_l_m):
                test_bed = np.mean(p_[o_ == 1.0] == t_[o_ == 1.0])
                accuracy_for_5 += 1 if test_bed == 1.0 else 0
        accuracy_for_5 /= test_5_data.data_size

        # Accuracy for N=5.
        accuracy_for_10 = 0
        for _ in range(test_10_data.test_iter):
            i_d, t_o, s_l, _, o_l_m, _ = test_10_data.get_test_sample_wlen()
            prediction = sess.run(final_result, feed_dict={
                dataset.input_data: i_d,
                dataset.sequence_length: s_l
            })
            for p_, t_, o_ in zip(prediction, t_o, o_l_m):
                test_bed = np.mean(p_[o_ == 1.0] == t_[o_ == 1.0])
                accuracy_for_10 += 1 if test_bed == 1.0 else 0
        accuracy_for_10 /= test_10_data.data_size

        tf.logging.info("Training Iteration %d/%d: N=5 Accuracy %.3f, N=10 Accuracy %.3f.\n", 0, epoch * iteration_per_epoch, accuracy_for_5, accuracy_for_10)

        loss = 0
        for e in range(epoch):
            for i in range(iteration_per_epoch):
                i_d, t_o, s_l, l_m, o_l_m, s_f = data.get_train_sample_wlen()

                fetches['summary_op'] = summary_op if i % FLAGS.summarize == 0 else no_summary
                fetches_ = sess.run(fetches, feed_dict={
                    dataset.input_data: i_d,
                    dataset.target_output: t_o,
                    dataset.sequence_length: s_l,
                    dataset.loss_mask: l_m,
                    dataset.original_loss_mask: o_l_m,
                    dataset.scale_factor: s_f,
                })
                loss += fetches_['loss']

                if i % FLAGS.summarize == 0:
                    summary_writer.add_summary(fetches_['summary_op'], fetches_['step'])

                if (e*iteration_per_epoch + i + 1) % report_interval == 0:

                    accuracy_for_5 = 0
                    for _ in range(test_5_data.test_iter):
                        i_d, t_o, s_l, _, o_l_m, _ = test_5_data.get_test_sample_wlen()
                        prediction = sess.run(final_result, feed_dict={
                            dataset.input_data: i_d,
                            dataset.sequence_length: s_l
                        })
                        for p_, t_, o_ in zip(prediction, t_o, o_l_m):
                            test_bed = np.mean(p_[o_ == 1.0] == t_[o_ == 1.0])
                            accuracy_for_5 += 1 if test_bed == 1.0 else 0
                    accuracy_for_5 /= test_5_data.data_size

                    # Accuracy for N=5.
                    accuracy_for_10 = 0
                    for _ in range(test_10_data.test_iter):
                        i_d, t_o, s_l, _, o_l_m, _ = test_10_data.get_test_sample_wlen()
                        prediction = sess.run(final_result, feed_dict={
                            dataset.input_data: i_d,
                            dataset.sequence_length: s_l
                        })
                        for p_, t_, o_ in zip(prediction, t_o, o_l_m):
                            test_bed = np.mean(p_[o_ == 1.0] == t_[o_ == 1.0])
                            accuracy_for_10 += 1 if test_bed == 1.0 else 0
                    accuracy_for_10 /= test_10_data.data_size

                    tf.logging.info("Training Iteration %d/%d: Avg loss %.3f,\nN=5 Accuracy %.3f, N=10 Accuracy %.3f.\n", e * iteration_per_epoch + i + 1,
                                    epoch * iteration_per_epoch, loss / report_interval, accuracy_for_5, accuracy_for_10)
                    loss = 0


def main(unused_argv):
    if not os.path.exists(os.path.join('info', FLAGS.name)):
        os.mkdir(os.path.join('info', FLAGS.name))

    start = time.time()
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    train(FLAGS.epoch, FLAGS.report_interval)
    print("Total time: %.2f min"%((time.time() - start)/60))


if __name__ == "__main__":
    tf.app.run()
