"""DAM Cores.

These modules create a DAM core. They take input, pass parameters to the memory
access module, and integrate the output of memory to form an output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from model.DAM import access
from model.DAM.util import layer_normalization

DAMState = collections.namedtuple('DAMState', ('access_output', 'access_state',
                                               'controller_state'))


class DAM(snt.RNNCore):
    """DAM core module.

    Contains controller and memory access module.
    """

    def __init__(self,
                 access_config,
                 controller_config,
                 other_config,
                 output_size,
                 clip_value=None,
                 name='DAM'):
        """Initializes the DAM core.

        Args:
          access_config: dictionary of access module configurations.
          controller_config: dictionary of controller (LSTM) module configurations.
          output_size: output dimension size of core.
          clip_value: clips controller and core output values to between
              `[-clip_value, clip_value]` if specified.
          name: module name (default 'DAM').

        Raises:
          TypeError: if direct_input_size is not None for any access module other
            than KeyValueMemory.
        """
        super(DAM, self).__init__(name=name)

        with self._enter_variable_scope():
            self._controller = snt.LSTM(**controller_config)
            self._access = access.MemoryAccess(num_memory_block=other_config['num_memory_block'], **access_config)

        self._access_output_size = np.prod(self._access.output_size.as_list())
        self._output_size = output_size
        self._clip_value = clip_value or 0

        self._act_fn_list = other_config['act_fn_list'] if 'act_fn_list' in other_config.keys() else []
        self._layer_size_list = other_config['layer_size_list'] if 'act_fn_list' in other_config.keys() else []

        self._keep_prob = other_config['keep_prob']

        self._output_size = tf.TensorShape([output_size])
        self._state_size = DAMState(
            access_output=self._access_output_size,
            access_state=self._access.state_size,
            controller_state=self._controller.state_size)

    def _clip_if_enabled(self, x):
        if self._clip_value > 0:
            return tf.clip_by_value(x, -self._clip_value, self._clip_value)
        else:
            return x

    def _build(self, inputs, prev_state):
        """Connects the DAM core into the graph.

        Args:
          inputs: Tensor input.
          prev_state: A `DAMState` tuple containing the fields `access_output`,
              `access_state` and `controller_state`. `access_state` is a 3-D Tensor
              of shape `[batch_size, num_reads, word_size]` containing read words.
              `access_state` is a tuple of the access module's state, and
              `controller_state` is a tuple of controller module's state.

        Returns:
          A tuple `(output, next_state)` where `output` is a tensor and `next_state`
          is a `DAMState` tuple containing the fields `access_output`,
          `access_state`, and `controller_state`.
        """

        prev_access_output = prev_state.access_output
        prev_access_state = prev_state.access_state
        prev_controller_state = prev_state.controller_state

        batch_flatten = snt.BatchFlatten()
        controller_input = tf.concat(
            [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

        controller_output, controller_state = self._controller(
            controller_input, prev_controller_state)

        controller_output = self._clip_if_enabled(controller_output)
        controller_state = tf.contrib.framework.nest.map_structure(self._clip_if_enabled, controller_state)

        controller_output = layer_normalization(controller_output)

        access_output, access_state = self._access(controller_output,
                                                   prev_access_state)

        controller_output = tf.nn.dropout(controller_output, self._keep_prob)

        output = tf.concat([controller_output, batch_flatten(access_output)], 1)

        for i, (act_fn, size) in enumerate(zip(self._act_fn_list, self._layer_size_list)):
            output = tf.layers.dense(output, size,
                                     activation=act_fn,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='projection_'+str(i),
                                     reuse=tf.AUTO_REUSE)

        output = snt.Linear(
            output_size=self._output_size.as_list()[0],
            name='output_linear')(output)
        output = self._clip_if_enabled(output)

        return output, DAMState(
            access_output=access_output,
            access_state=access_state,
            controller_state=controller_state)

    def initial_state(self, batch_size, dtype=tf.float32):
        return DAMState(
            controller_state=self._controller.initial_state(batch_size, dtype),
            access_state=self._access.initial_state(batch_size, dtype),
            access_output=tf.zeros(
                [batch_size] + self._access.output_size.as_list(), dtype))

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size
