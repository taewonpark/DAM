"""DNC access modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sonnet as snt
import tensorflow as tf

from model.DAM import addressing
from model.DAM import util

AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'usage', 'attentive_gate'))


def _erase_and_write(memory, address, reset_weights, values):
    """Module to erase and write in the external memory.

    Erase operation:
      M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

    Add operation:
      M_t(i) = M_t'(i) + w_t(i) * a_t

    where e are the reset_weights, w the write weights and a the values.

    Args:
      memory: 4-D tensor of shape `[batch_size, num_memory_blocks, memory_size, word_size]`.
      address: 4-D tensor `[batch_size, num_memory_blocks, num_writes, memory_size]`.
      reset_weights: 4-D tensor `[batch_size, num_memory_blocks, num_writes, word_size]`.
      values: 4-D tensor `[batch_size, num_memory_blocks, num_writes, word_size]`.

    Returns:
      3-D tensor of shape `[batch_size, num_writes, word_size]`.
    """
    with tf.name_scope('erase_memory', values=[memory, address, reset_weights]):
        expand_address = tf.expand_dims(address, 4)
        reset_weights = tf.expand_dims(reset_weights, 3)
        weighted_resets = expand_address * reset_weights
        weighted_resets = tf.transpose(weighted_resets, [0, 2, 1, 3, 4])
        reset_gate = util.reduce_prod(1 - weighted_resets, 1)
        memory *= reset_gate

    with tf.name_scope('additive_write', values=[memory, address, values]):
        add_matrix = tf.matmul(address, values, adjoint_a=True)
        memory += add_matrix

    return memory


class MemoryAccess(snt.RNNCore):
    """Access module of the Differentiable Neural Computer.

    This memory module supports multiple read and write heads. It makes use of:

    *   `addressing.FreenessAllocator` for keeping track of memory usage, where
        usage increase when a memory location is written to, and decreases when
        memory is read from that the controller says can be freed.

    Write-address selection is done by an interpolation between content-based
    lookup and using unused memory.

    Read-address selection is done by an interpolation of content-based lookup
    and following the link graph in the forward or backwards read direction.
    """

    def __init__(self,
                 num_memory_block=2,
                 memory_size=128,
                 word_size=20,
                 num_reads=1,
                 num_writes=1,
                 clip=False,
                 name='memory_access'):
        """Creates a MemoryAccess module.

        Args:
          num_memory_block: The number of memory blocks.
          memory_size: The number of memory slots (N in the DNC paper).
          word_size: The width of each memory slot (W in the DNC paper)
          num_reads: The number of read heads (R in the DNC paper).
          num_writes: The number of write heads (fixed at 1 in the paper).
          name: The name of the module.
        """
        super(MemoryAccess, self).__init__(name=name)
        self._num_memory_block = num_memory_block
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes
        self._clip = clip

        self._write_content_weights_mod = addressing.CosineWeights(
            num_writes, word_size, name='write_content_weights')
        self._read_content_weights_mod = addressing.CosineWeights(
            num_reads, word_size, name='read_content_weights')

        self._freeness = addressing.Freeness(num_memory_block, memory_size)

        self._interface_size, self._range_list = self._bulid_var()

    def _bulid_var(self):
        parameter_size_list = [
            self._num_memory_block * self._num_writes * self._word_size,  # write_vectors
            self._num_memory_block * self._num_writes * self._word_size,  # erase_vectors
            self._num_memory_block * self._num_reads,  # free_gate
            self._num_memory_block * self._num_writes,  # allocation_gate
            self._num_memory_block * self._num_writes,  # write_gate
            self._num_memory_block * self._num_writes * self._word_size,  # write_keys
            self._num_memory_block * self._num_writes,  # write_strengths
            self._num_memory_block * self._num_reads * self._word_size,  # read_keys
            self._num_memory_block * self._num_reads,  # read_strengths
            self._num_memory_block * self._num_reads  # which read vector
        ]
        interface_size = sum(parameter_size_list)

        range_list = [0]
        for size in parameter_size_list:
            range_list.append(range_list[-1]+size)

        return interface_size, range_list

    def _build(self, inputs, prev_state):
        """Connects the MemoryAccess module into the graph.

        Args:
          inputs: tensor of shape `[batch_size, input_size]`. This is used to
              control this access module.
          prev_state: Instance of `AccessState` containing the previous state.

        Returns:
          A tuple `(output, next_state)`, where `output` is a tensor of shape
          `[batch_size, num_reads, word_size]`, and `next_state` is the new
          `AccessState` named tuple at the current time t.
        """
        control = self._read_inputs(inputs)

        # Update usage using inputs['free_gate'] and previous read & write weights.
        usage = self._freeness(
            write_weights=prev_state.write_weights,
            free_gate=control['free_gate'],
            read_weights=prev_state.read_weights,
            prev_usage=prev_state.usage)

        # Write to memory.
        write_weights = self._write_weights(control, prev_state.memory, usage)
        memory = _erase_and_write(
            prev_state.memory,
            address=write_weights,
            reset_weights=control['erase_vectors'],
            values=control['write_vectors'])

        # Read from memory.
        read_weights = self._read_weights(
            control,
            memory=memory)
        read_weights = tf.multiply(read_weights, control['attentive_gate'])
        read_words = tf.matmul(read_weights, memory)
        read_words = tf.reduce_sum(read_words, axis=1)

        return (read_words, AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            usage=usage,
            attentive_gate=control['attentive_gate']))

    def _read_inputs(self, inputs):
        """Applies transformations to `inputs` to get control for this module."""

        def _alloc_1d(input, num_memory_block, first_dim, name, activation=None):
            vector = tf.reshape(input, [-1, num_memory_block, first_dim], name=name)
            if activation is not None:
                vector = activation(vector, name=name + '_activation')
            return vector

        def _alloc_2d(input, num_memory_block, first_dim, second_dim, name, activation=None):
            vector = tf.reshape(input, [-1, num_memory_block, first_dim, second_dim], name=name)
            if activation is not None:
                vector = activation(vector, name=name + '_activation')
            return vector

        interface_vector = tf.layers.dense(inputs, self._interface_size,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           name='interface_vector',
                                           reuse=tf.AUTO_REUSE)
        interface_vector = util.clip_if_enabled(interface_vector) if self._clip else interface_vector

        write_vectors = _alloc_2d(interface_vector[:, self._range_list[0]:self._range_list[1]],
                                  self._num_memory_block, self._num_writes, self._word_size, 'write_vectors')

        erase_vectors = _alloc_2d(interface_vector[:, self._range_list[1]:self._range_list[2]],
                                  self._num_memory_block, self._num_writes, self._word_size, 'erase_vectors', tf.sigmoid)

        free_gate = _alloc_1d(interface_vector[:, self._range_list[2]:self._range_list[3]],
                              self._num_memory_block, self._num_reads, 'free_gate', tf.sigmoid)

        allocation_gate = _alloc_1d(interface_vector[:, self._range_list[3]:self._range_list[4]],
                                    self._num_memory_block, self._num_writes, 'allocation_gate', tf.sigmoid)

        write_gate = _alloc_1d(interface_vector[:, self._range_list[4]:self._range_list[5]],
                               self._num_memory_block, self._num_writes, 'write_gate', tf.sigmoid)
        write_keys = _alloc_2d(interface_vector[:, self._range_list[5]:self._range_list[6]],
                               self._num_memory_block, self._num_writes, self._word_size, 'write_keys')
        write_strengths = _alloc_1d(interface_vector[:, self._range_list[6]:self._range_list[7]],
                                    self._num_memory_block, self._num_writes, 'write_strengths')

        read_keys = _alloc_2d(interface_vector[:, self._range_list[7]:self._range_list[8]],
                              self._num_memory_block, self._num_reads, self._word_size, 'read_keys')
        read_strengths = _alloc_1d(interface_vector[:, self._range_list[8]:self._range_list[9]],
                                   self._num_memory_block, self._num_reads, 'read_strengths')

        attentive_gate = _alloc_2d(interface_vector[:, self._range_list[9]:self._range_list[10]],
                                   self._num_memory_block, self._num_reads, 1, 'attentive_gate')

        result = {
            'read_content_keys': read_keys,
            'read_content_strengths': read_strengths,
            'write_content_keys': write_keys,
            'write_content_strengths': write_strengths,
            'write_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gate': free_gate,
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'attentive_gate': tf.nn.softmax(attentive_gate, axis=1),
        }

        return result

    def _write_weights(self, inputs, memory, usage):
        """Calculates the memory locations to write to.

        This uses a combination of content-based lookup and finding an unused
        location in memory, for each write head.

        Args:
          inputs: Collection of inputs to the access module, including controls for
              how to chose memory writing, such as the content to look-up and the
              weighting between content-based and allocation-based addressing.
          memory: A tensor of shape  `[batch_size, num_memory_blocks, memory_size, word_size]`
              containing the current memory contents.
          usage: Current memory usage, which is a tensor of shape `[batch_size, num_memory_blocks,
              memory_size]`, used for allocation-based addressing.

        Returns:
          tensor of shape `[batch_size, num_memory_blocks, num_writes, memory_size]` indicating where
              to write to (if anywhere) for each write head.
        """
        with tf.name_scope('write_weights', values=[inputs, memory, usage]):
            # c_t^{w, i} - The content-based weights for each write head.
            write_content_weights = self._write_content_weights_mod(
                memory, inputs['write_content_keys'],
                inputs['write_content_strengths'])

            # a_t^i - The allocation weights for each write head.
            write_allocation_weights = self._freeness.write_allocation_weights(
                usage=usage,
                write_gates=(inputs['allocation_gate'] * inputs['write_gate']),
                num_writes=self._num_writes)

            # Expands gates over memory locations.
            allocation_gate = tf.expand_dims(inputs['allocation_gate'], -1)
            write_gate = tf.expand_dims(inputs['write_gate'], -1)

            # w_t^{w, i} - The write weightings for each write head.
            return write_gate * (allocation_gate * write_allocation_weights +
                                 (1 - allocation_gate) * write_content_weights)

    def _read_weights(self, inputs, memory):
        """Calculates read weights for each read head.

        The read weights are a combination of following the link graphs in the
        forward or backward directions from the previous read position, and doing
        content-based lookup. The interpolation between these different modes is
        done by `inputs['read_mode']`.

        Args:
          inputs: Controls for this access module. This contains the content-based
              keys to lookup, and the weightings for the different read modes.
          memory: A tensor of shape `[batch_size, num_memory_blocks, memory_size, word_size]`
              containing the current memory contents to do content-based lookup.
          name: string, 'left' or 'right'.

        Returns:
          A tensor of shape `[batch_size, num_memory_blocks, num_reads, memory_size]` containing the
          read weights for each read head.
        """
        with tf.name_scope(
                'read_weights', values=[inputs, memory]):
            # c_t^{r, i} - The content weightings for each read head.
            read_weights = self._read_content_weights_mod(
                memory, inputs['read_content_keys'], inputs['read_content_strengths'])

            return read_weights

    @property
    def state_size(self):
        """Returns a tuple of the shape of the state tensors."""
        return AccessState(
            memory=tf.TensorShape([self._num_memory_block, self._memory_size, self._word_size]),
            read_weights=tf.TensorShape([self._num_memory_block, self._num_reads, self._memory_size]),
            write_weights=tf.TensorShape([self._num_memory_block, self._num_writes, self._memory_size]),
            usage=self._freeness.state_size,
            attentive_gate=tf.TensorShape([self._num_memory_block, self._num_reads, 1]))

    @property
    def output_size(self):
        """Returns the output shape."""
        return tf.TensorShape([self._num_reads, self._word_size])
