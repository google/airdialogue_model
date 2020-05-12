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
"""This module contains a modified version of tensorflow basic decoder for AirDialogue."""




import collections

import tensorflow as tf

from tensorflow.contrib.seq2seq import Decoder

from tensorflow.layers import Layer

from rnn_decoder import helper as helper_py


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
  pass


class BasicDecoder(Decoder):
  """Basic sampling decoder."""

  def __init__(self,
               cell,
               helper,
               initial_state,
               encoder_outputs,
               turn_points,
               output_layer=None,
               aux_hidden_state=None):
    """Initialize BasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      encoder_outputs: the output of the encoder
      turn_points: points where conversations switch party
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior to
        storing the result or sampling.
      aux_hidden_state: hidden embeddings of context information

    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None and
        not isinstance(output_layer, Layer)):
      raise TypeError("output_layer must be a Layer, received: %s" %
                      type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer
    self.encoder_outputs = encoder_outputs
    self.turn_points = turn_points
    self._aux_hidden_state = aux_hidden_state

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = tf.nest.map_structure(
          lambda s: tf.TensorShape([None]).concatenate(s), size)
      layer_output_shape = self._output_layer.compute_output_shape(  # pylint: disable=protected-access
          output_shape_with_unknown_batch)
      return tf.nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        sample_id=tf.TensorShape([]))

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and int32 (the id)
    dtype = tf.nest.flatten(self._initial_state)[0].dtype
    return BasicDecoderOutput(
        tf.nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        tf.int32)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with tf.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      if isinstance(state, tuple):
        bs = tf.shape(state[0])[0]
        embs = tf.shape(state[0])[1]
        weight1 = self.turn_points[:, time]
        weight1 = tf.tile(weight1, [embs])
        weight1 = tf.reshape(weight1, [bs, embs])
        state_list = list(state)
        for i in range(len(state_list)):
          state_list[i] = tf.multiply(state_list[i], 1 - weight1) + tf.multiply(
              self.encoder_outputs[:, time, i * embs:(i + 1) * embs], weight1)
        new_state = tuple(state_list)
      else:
        bs = tf.shape(state)[0]
        embs = tf.shape(state)[1]
        weight1 = self.turn_points[:, time]
        weight1 = tf.tile(weight1, [embs])
        weight1 = tf.reshape(weight1, [bs, embs])
        new_state = tf.multiply(state, 1 - weight1) + tf.multiply(
            self.encoder_outputs[:, time, :], weight1)
      cell_outputs, cell_state = self._cell(inputs, new_state)

      if self._output_layer is not None:
        concat = cell_outputs
        cell_outputs = self._output_layer(concat)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)
