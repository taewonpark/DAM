"""DAM addressing modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

from model.DAM_test import util

# Ensure values are greater than epsilon to avoid numerical instability.
_EPSILON = 1e-6


def _vector_norms(m):
  squared_norms = tf.reduce_sum(m * m, axis=3, keepdims=True)
  return tf.sqrt(squared_norms + _EPSILON)


def weighted_softmax(activations, strengths, strengths_op):
  """Returns softmax over activations multiplied by positive strengths.

  Args:
    activations: A tensor of shape `[batch_size, num_memory_blocks, num_heads, memory_size]`, of
      activations to be transformed. Softmax is taken over the last dimension.
    strengths: A tensor of shape `[batch_size, num_memory_blocks, num_heads]` containing strengths to
      multiply by the activations prior to the softmax.
    strengths_op: An operation to transform strengths before softmax.

  Returns:
    A tensor of same shape as `activations` with weighted softmax applied.
  """
  transformed_strengths = tf.expand_dims(strengths_op(strengths), axis=-1)
  sharp_activations = activations * transformed_strengths
  return tf.nn.softmax(sharp_activations, axis=-1)


class CosineWeights(snt.AbstractModule):
  """Cosine-weighted attention.

  Calculates the cosine similarity between a query and each word in memory, then
  applies a weighted softmax to return a sharp distribution.
  """

  def __init__(self,
               num_heads,
               word_size,
               strength_op=tf.nn.softplus,
               name='cosine_weights'):
    """Initializes the CosineWeights module.

    Args:
      num_heads: number of memory heads.
      word_size: memory word size.
      strength_op: operation to apply to strengths (default is tf.nn.softplus).
      name: module name (default 'cosine_weights')
    """
    super(CosineWeights, self).__init__(name=name)
    self._num_heads = num_heads
    self._word_size = word_size
    self._strength_op = strength_op

  def _build(self, memory, keys, strengths):
    """Connects the CosineWeights module into the graph.

    Args:
      memory: A 4-D tensor of shape `[batch_size, num_memory_blocks, memory_size, word_size]`.
      keys: A 4-D tensor of shape `[batch_size, num_memory_blocks, num_heads, word_size]`.
      strengths: A 3-D tensor of shape `[batch_size, num_memory_blocks, num_heads]`.

    Returns:
      Weights tensor of shape `[batch_size, num_memory_blocks, num_heads, memory_size]`.
    """
    # Calculates the inner product between the query vector and words in memory.
    dot = tf.matmul(keys, memory, adjoint_b=True)

    # Outer product to compute denominator (euclidean norm of query and memory).
    memory_norms = _vector_norms(memory)
    key_norms = _vector_norms(keys)
    norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)

    # Calculates cosine similarity between the query vector and words in memory.
    similarity = dot / (norm + _EPSILON)

    return weighted_softmax(similarity, strengths, self._strength_op)


class Freeness(snt.RNNCore):
  """Memory usage that is increased by writing and decreased by reading.

  This module is a pseudo-RNNCore whose state is a tensor with values in
  the range [0, 1] indicating the usage of each of `memory_size` memory slots.

  The usage is:

  *   Increased by writing, where usage is increased towards 1 at the write
      addresses.
  *   Decreased by reading, where usage is decreased after reading from a
      location when free_gate is close to 1.

  The function `write_allocation_weights` can be invoked to get free locations
  to write to for a number of write heads.
  """

  def __init__(self, num_memory_block, memory_size, name='freeness'):
    """Creates a Freeness module.

    Args:
      memory_size: Number of memory slots.
      name: Name of the module.
    """
    super(Freeness, self).__init__(name=name)
    self._num_memory_block = num_memory_block
    self._memory_size = memory_size

  def _build(self, write_weights, free_gate, read_weights, prev_usage):
    """Calculates the new memory usage u_t.

    Memory that was written to in the previous time step will have its usage
    increased; memory that was read from and the controller says can be "freed"
    will have its usage decreased.

    Args:
      write_weights: tensor of shape `[batch_size, num_memory_blocks, num_writes,
          memory_size]` giving write weights at previous time step.
      free_gate: tensor of shape `[batch_size, num_memory_blocks, num_reads]` which indicates
          which read heads read memory that can now be freed.
      read_weights: tensor of shape `[batch_size, num_memory_blocks, num_reads,
          memory_size]` giving read weights at previous time step.
      prev_usage: tensor of shape `[batch_size, num_memory_blocks, memory_size]` giving
          usage u_{t - 1} at the previous time step, with entries in range
          [0, 1].

    Returns:
      tensor of shape `[batch_size, num_memory_blocks, memory_size]` representing updated memory
      usage.
    """
    # Calculation of usage is not differentiable with respect to write weights.
    write_weights = tf.stop_gradient(write_weights)
    usage = self._usage_after_write(prev_usage, write_weights)
    usage = self._usage_after_read(usage, free_gate, read_weights)
    return usage

  def write_allocation_weights(self, usage, write_gates, num_writes):
    """Calculates freeness-based locations for writing to.

    This finds unused memory by ranking the memory locations by usage, for each
    write head. (For more than one write head, we use a "simulated new usage"
    which takes into account the fact that the previous write head will increase
    the usage in that area of the memory.)

    Args:
      usage: A tensor of shape `[batch_size, num_memory_blocks, memory_size]` representing
          current memory usage.
      write_gates: A tensor of shape `[batch_size, num_memory_blocks, num_writes]` with values in
          the range [0, 1] indicating how much each write head does writing
          based on the address returned here (and hence how much usage
          increases).
      num_writes: The number of write heads to calculate write weights for.

    Returns:
      tensor of shape `[batch_size, num_memory_blocks, num_writes, memory_size]` containing the
          freeness-based write locations. Note that this isn't scaled by
          `write_gate`; this scaling must be applied externally.
    """
    with tf.name_scope('write_allocation_weights'):
      # expand gatings over memory locations
      write_gates = tf.expand_dims(write_gates, -1)

      allocation_weights = []
      for i in range(num_writes):
        allocation_weights.append(self._allocation(usage))
        # update usage to take into account writing to this new allocation
        usage += ((1 - usage) * write_gates[:, i, :, :] * allocation_weights[i])

      # Pack the allocation weights for the write heads into one tensor.
      return tf.stack(allocation_weights, axis=2)

  def _usage_after_write(self, prev_usage, write_weights):
    """Calcualtes the new usage after writing to memory.

    Args:
      prev_usage: tensor of shape `[batch_size, num_memory_blocks, memory_size]`.
      write_weights: tensor of shape `[batch_size, num_memory_blocks, num_writes, memory_size]`.

    Returns:
      New usage, a tensor of shape `[batch_size, num_memory_blocks, memory_size]`.
    """
    with tf.name_scope('usage_after_write'):
      # Calculate the aggregated effect of all write heads
      write_weights = tf.transpose(write_weights, [0, 2, 1, 3])
      write_weights = 1 - util.reduce_prod(1 - write_weights, 1)
      return prev_usage + (1 - prev_usage) * write_weights

  def _usage_after_read(self, prev_usage, free_gate, read_weights):
    """Calcualtes the new usage after reading and freeing from memory.

    Args:
      prev_usage: tensor of shape `[batch_size, num_memory_blocks, memory_size]`.
      free_gate: tensor of shape `[batch_size, num_memory_blocks, num_reads]` with entries in the
          range [0, 1] indicating the amount that locations read from can be
          freed.
      read_weights: tensor of shape `[batch_size, num_memory_blocks, num_reads, memory_size]`.

    Returns:
      New usage, a tensor of shape `[batch_size, num_memory_blocks, memory_size]`.
    """
    with tf.name_scope('usage_after_read'):
      free_gate = tf.expand_dims(free_gate, -1)
      free_read_weights = free_gate * read_weights
      free_read_weights = tf.transpose(free_read_weights, [0, 2, 1, 3])
      phi = util.reduce_prod(1 - free_read_weights, 1, name='phi')
      return prev_usage * phi

  def _allocation(self, usage):
    r"""Computes allocation by sorting `usage`.

    This corresponds to the value a = a_t[\phi_t[j]] in the paper.

    Args:
      usage: tensor of shape `[batch_size, num_memory_blocks, memory_size]` indicating current
          memory usage. This is equal to u_t in the paper when we only have one
          write head, but for multiple write heads, one should update the usage
          while iterating through the write heads to take into account the
          allocation returned by this function.

    Returns:
      Tensor of shape `[batch_size, num_memory_blocks, memory_size]` corresponding to allocation.
    """
    with tf.name_scope('allocation'):
      # Ensure values are not too small prior to cumprod.
      usage = _EPSILON + (1 - _EPSILON) * usage

      nonusage = 1 - usage
      sorted_nonusage, indices = tf.nn.top_k(
        nonusage, k=self._memory_size, name='sort')
      sorted_usage = 1 - sorted_nonusage
      prod_sorted_usage = tf.cumprod(sorted_usage, axis=2, exclusive=True)
      sorted_allocation = sorted_nonusage * prod_sorted_usage
      inverse_indices = util.batch_invert_permutation(indices)

      # This final line "unsorts" sorted_allocation, so that the indexing
      # corresponds to the original indexing of `usage`.
      return util.batch_gather(sorted_allocation, inverse_indices)

  @property
  def state_size(self):
    """Returns the shape of the state tensor."""
    return tf.TensorShape([self._num_memory_block, self._memory_size])
