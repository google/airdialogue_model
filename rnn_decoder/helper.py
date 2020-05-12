# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""this file contains helper."""

import abc
import six

from tensorflow.contrib.rnn import transpose_batch_time

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest


def _unstack_ta(inp):
  return tf.TensorArray(
      dtype=inp.dtype, size=tf.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)


@six.add_metaclass(abc.ABCMeta)
class Helper(object):
  """Interface for implementing sampling in seq2seq decoders.
  Helper instances are used by `BasicDecoder`.
  """

  @abc.abstractproperty
  def batch_size(self):
    """Batch size of tensor returned by `sample`.
    Returns a scalar int32 tensor.
    """
    raise NotImplementedError("batch_size has not been implemented")

  @abc.abstractmethod
  def initialize(self, name=None):
    """Returns `(initial_finished, initial_inputs)`."""
    pass

  @abc.abstractmethod
  def sample(self, time, outputs, state, name=None):
    """Returns `sample_ids`."""
    pass

  @abc.abstractmethod
  def next_inputs(self, time, outputs, state, sample_ids, name=None):
    """Returns `(finished, next_inputs, next_state)`."""
    pass


class TrainingHelper(Helper):
  """A helper for use during training.
  Only reads inputs.
  Returned sample_ids are the argmax of the RNN output logits.
  """

  def __init__(self, inputs, sequence_length, time_major=False, name=None):
    """Initializer.

    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      name: Name scope for any created operations.
    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    with tf.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
      inputs = tf.convert_to_tensor(inputs, name="inputs")
      if not time_major:
        inputs = nest.map_structure(transpose_batch_time, inputs)

      self._input_tas = nest.map_structure(_unstack_ta, inputs)
      self._sequence_length = tf.convert_to_tensor(
          sequence_length, name="sequence_length")
      if self._sequence_length.get_shape().ndims != 1:
        raise ValueError(
            "Expected sequence_length to be a vector, but received shape: %s" %
            self._sequence_length.get_shape())

      self._zero_inputs = nest.map_structure(
          lambda inp: tf.zeros_like(inp[0, :]), inputs)

      self._batch_size = tf.size(sequence_length)

  @property
  def batch_size(self):
    return self._batch_size

  def initialize(self, name=None):
    with tf.name_scope(name, "TrainingHelperInitialize"):
      finished = tf.equal(0, self._sequence_length)
      return (finished, self._zero_inputs)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    with tf.name_scope(name, "TrainingHelperSample", [time, outputs]):
      sample_ids = tf.cast(
          tf.argmax(outputs, axis=-1), tf.dtypes.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    """next_inputs_fn for TrainingHelper."""
    with tf.name_scope(name, "TrainingHelperNextInputs",
                        [time, outputs, state]):
      # next_time = time + 1
      next_time = time +1

      finished = (next_time >= self._sequence_length)
      all_finished = tf.reduce_all(finished)
      def read_from_ta(inp):
        return inp.read(next_time-1)
      next_inputs = tf.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: nest.map_structure(read_from_ta, self._input_tas))
      return (finished, next_inputs, state)


