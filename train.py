#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sonnet as snt
import os

from model.DAM import dam
from model.DNC import dnc
from loader import BAbIBatchGenerator, BAbIData
from loader import Copy, AssociativeRecall, RepresentationRecall, AlgorithmicData

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("embedding_size", 64, "Size of embedding.")
tf.flags.DEFINE_integer("hidden_size", 128, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_address_size", 64, "The number of memory slots.")
tf.flags.DEFINE_integer("memory_length_size", 24, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 1, "Number of memory read heads.")
tf.flags.DEFINE_float("keep_prob", 0.9, "Keep probability for bypass dropout")
tf.flags.DEFINE_integer("num_memory_blocks", 2, "Number of memory blocks.")

# Optimizer parameters.
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("momentum", 0.9, "Optimizer momentum.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10, "Epsilon used for RMSProp optimizer.")
tf.flags.DEFINE_integer("clip_value", 20, "Maximum absolute value of controller and model outputs.")

# Model selection.
tf.flags.DEFINE_boolean("dam", True, "Whether dam or not.")

# WMR loss parameters.
tf.flags.DEFINE_float("p_re", 0.0, "Reproducing Probability.")

# Task Configuration.
tf.flags.DEFINE_string("mode", "Copy", "Task mode")
tf.flags.DEFINE_integer("N", 4, "Number of sub-parts.")
tf.flags.DEFINE_integer("bit_w", 8, "Size of bit.")
tf.flags.DEFINE_integer("num_bit", 8, "Number of bit.")
tf.flags.DEFINE_integer("item_bit", 3, "Number of bit per item.")
tf.flags.DEFINE_integer("min_length", 8, "Minimum length")
tf.flags.DEFINE_integer("max_length", 32, "Maximum length")

# Training options.
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("epoch", 50, "Number of epochs to train for.")
tf.flags.DEFINE_integer("training_iteration", 10000, "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 500, "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_integer("checkpoint_interval", 1, "Checkpointing step interval.")
tf.flags.DEFINE_string("name", "model", "Name of training model.")


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
    clip_value = FLAGS.clip_value

    if FLAGS.dam:
        core = dam.DAM(access_config, controller_config, other_config, output_size, clip_value)
    else:
        core = dnc.DNC(access_config, controller_config, output_size, clip_value)

    print("Core: " + str(core))

    initial_state = core.initial_state(FLAGS.batch_size)
    output_sequence, _ = tf.nn.dynamic_rnn(
        cell=core,
        inputs=input_data,
        sequence_length=sequence_length,
        time_major=False,
        initial_state=initial_state)

    return output_sequence


def train(epoch, report_interval):
    """Trains the model and periodically reports the loss."""

    if FLAGS.mode == "bAbI":
        train_data = BAbIBatchGenerator(FLAGS.batch_size, FLAGS.p_re, shuffle=True)
        dataset = BAbIData(FLAGS.batch_size, train_data.input_size, train_data.output_size, FLAGS.embedding_size)
        output_logits = run_model(dataset.processed_input_data, dataset.sequence_length, train_data.output_size)
    else:
        if FLAGS.mode == "Copy":
            train_data = Copy(FLAGS.batch_size, FLAGS.p_re, FLAGS.bit_w, FLAGS.min_length, FLAGS.max_length)
        elif FLAGS.mode == "AssociativeRecall":
            train_data = AssociativeRecall(FLAGS.batch_size, FLAGS.p_re, FLAGS.bit_w, FLAGS.item_bit, FLAGS.min_length, FLAGS.max_length)
        elif FLAGS.mode == 'RepresentationRecall':
            train_data = RepresentationRecall(FLAGS.batch_size, FLAGS.N, FLAGS.bit_w, FLAGS.num_bit, FLAGS.min_length, FLAGS.max_length)
        dataset = AlgorithmicData(FLAGS.batch_size, train_data.input_size, train_data.output_size)
        output_logits = run_model(dataset.input_data, dataset.sequence_length, train_data.output_size)

    print("Memory Refreshing Probability: " + str(FLAGS.p_re))
    print("Train Data Generator: " + str(train_data))
    print("Dataset: " + str(dataset))

    train_loss = dataset.cost(output_logits)
    original_loss = dataset.check_cost(output_logits)

    tf.summary.scalar("loss", original_loss)
    summary_op = tf.summary.merge_all()

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
    save_steps = int(train_data.data_size/FLAGS.batch_size)*FLAGS.checkpoint_interval if ('bAbI' in FLAGS.mode) else FLAGS.training_iteration

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
            'loss': original_loss,
            'summary_op': summary_op,
            'step': global_step,
        }

        summary_writer = tf.summary.FileWriter(os.path.join('info', FLAGS.name, "log"), graph=tf.get_default_graph())

        if 'bAbI' in FLAGS.mode:
            iteration_per_epoch = int(train_data.data_size / FLAGS.batch_size)
        else:
            iteration_per_epoch = int(FLAGS.training_iteration / epoch)

        loss = 0
        for e in range(epoch):
            for i in range(iteration_per_epoch):
                i_d, t_o, s_l, l_m, o_l_m = next(train_data)

                fetches_ = sess.run(fetches, feed_dict={
                    dataset.input_data: i_d,
                    dataset.target_output: t_o,
                    dataset.sequence_length: s_l,
                    dataset.loss_mask: l_m,
                    dataset.original_loss_mask: o_l_m
                })
                loss += fetches_['loss']

                summary_writer.add_summary(fetches_['summary_op'], fetches_['step'])

                if (e*iteration_per_epoch + i + 1) % report_interval == 0:
                    if 'bAbI' in FLAGS.mode:
                        tf.logging.info("Training Iteration %d/%d: Avg training loss %f.\n",
                                        e * iteration_per_epoch + i + 1, epoch * iteration_per_epoch,
                                        loss / report_interval)
                    else:
                        tf.logging.info("Training Iteration %d/%d: Avg training accuracy %f.\n",
                                        e * iteration_per_epoch + i + 1, epoch * iteration_per_epoch,
                                        np.exp(- loss / report_interval))
                    loss = 0


def main(unused_argv):
    if not os.path.exists(os.path.join('info', FLAGS.name)):
        os.mkdir(os.path.join('info', FLAGS.name))

    tf.logging.set_verbosity(3)  # Print INFO log messages.
    train(FLAGS.epoch, FLAGS.report_interval)


if __name__ == "__main__":
    tf.app.run()
