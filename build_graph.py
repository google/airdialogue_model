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

"""procedures to build graph."""
import tensorflow.compat.v1 as tf
from tensorflow.contrib import seq2seq
import model_helper
from rnn_decoder import basic_decoder
from rnn_decoder import helper as help_py
from utils import dialogue_utils
from utils import misc_utils as utils
from utils import vocab_utils


def _build_encoder_cell(model,
                        hparams,
                        num_layers,
                        num_residual_layers,
                        base_gpu=0,
                        all_layer_outputs=False):
  """multi rnn cell for the seq2seq encoder."""
  return model_helper.create_rnn_cell(
      num_units=hparams.num_units,
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      dropout=hparams.dropout,
      num_gpus=hparams.num_gpus,
      mode=model.mode,
      base_gpu=base_gpu,
      single_cell_fn=model.single_cell_fn,
      all_layer_outputs=all_layer_outputs)


def _build_encoder(model, encoder_emb_inp, hparams):
  """Build an seq2seq encoder."""
  num_layers = hparams.num_layers
  num_residual_layers = hparams.num_residual_layers

  iterator = model.iterator

  with tf.variable_scope("encoder") as scope:
    dtype = scope.dtype

    # Encoder_outpus: [max_time, batch_size, num_units]
    utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                    (num_layers, num_residual_layers))
    cell = _build_encoder_cell(
        model,
        hparams,
        num_layers,
        num_residual_layers,
        base_gpu=model.global_gpu_num,
        all_layer_outputs=True)
    model.global_gpu_num += num_layers

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        cell,
        encoder_emb_inp,
        dtype=dtype,
        sequence_length=iterator.dialogue_len,
        time_major=False,
        swap_memory=True)
  return encoder_outputs, encoder_state


def _build_decoder_cell(model, hparams, encoder_state, base_gpu):
  """multi rnn cell for the seq2seq decoder."""

  num_layers = hparams.num_layers
  num_residual_layers = hparams.num_residual_layers
  cell = model_helper.create_rnn_cell(
      num_units=hparams.num_units,
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      dropout=hparams.dropout,
      num_gpus=hparams.num_gpus,
      mode=model.mode,
      single_cell_fn=model.single_cell_fn,
      base_gpu=base_gpu)

  # For beam search, we need to replicate encoder infos beam_width times
  if model.mode == tf.estimator.ModeKeys.PREDICT and hparams.beam_width > 0:
    decoder_initial_state = seq2seq.tile_batch(
        encoder_state, multiplier=hparams.beam_width)
  else:
    decoder_initial_state = encoder_state

  return cell, decoder_initial_state


def _build_decoder(model, encoder_outputs, encoder_state, hparams, start_token,
                   end_token, output_layer, aux_hidden_state):
  """build decoder for the seq2seq model."""

  iterator = model.iterator

  start_token_id = tf.cast(
      model.vocab_table.lookup(tf.constant(start_token)), tf.int32)
  end_token_id = tf.cast(
      model.vocab_table.lookup(tf.constant(end_token)), tf.int32)

  start_tokens = tf.fill([model.batch_size], start_token_id)
  end_token = end_token_id

  ## Decoder.
  with tf.variable_scope("decoder") as decoder_scope:
    cell, decoder_initial_state = _build_decoder_cell(
        model, hparams, encoder_state, base_gpu=model.global_gpu_num)
    model.global_gpu_num += hparams.num_layers
    # ## Train or eval

    decoder_emb_inp = tf.nn.embedding_lookup(model.embedding_decoder,
                                             iterator.target)
    # Helper
    helper_train = help_py.TrainingHelper(
        decoder_emb_inp, iterator.dialogue_len, time_major=False)

    # Decoder
    my_decoder_train = basic_decoder.BasicDecoder(
        cell,
        helper_train,
        decoder_initial_state,
        encoder_outputs,
        iterator.turns,
        output_layer=output_layer,
        aux_hidden_state=aux_hidden_state)

    # Dynamic decoding
    outputs_train, _, _ = seq2seq.dynamic_decode(
        my_decoder_train,
        output_time_major=False,
        swap_memory=True,
        scope=decoder_scope)

    sample_id_train = outputs_train.sample_id
    logits_train = outputs_train.rnn_output
    ## Inference
    # else:

    beam_width = hparams.beam_width
    length_penalty_weight = hparams.length_penalty_weight

    if model.mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
      my_decoder_infer = seq2seq.BeamSearchDecoder(
          cell=cell,
          embedding=model.embedding_decoder,
          start_tokens=start_tokens,
          end_token=end_token,
          initial_state=decoder_initial_state,
          beam_width=beam_width,
          output_layer=output_layer,
          length_penalty_weight=length_penalty_weight)
    else:
      # Helper
      if model.mode in dialogue_utils.self_play_modes:
        helper_infer = seq2seq.SampleEmbeddingHelper(
            model.embedding_decoder, start_tokens, end_token)
      else:  # inference
        helper_infer = seq2seq.GreedyEmbeddingHelper(
            model.embedding_decoder, start_tokens, end_token)

      # Decoder
      my_decoder_infer = seq2seq.BasicDecoder(
          cell,
          helper_infer,
          decoder_initial_state,
          output_layer=output_layer  # applied per timestep
      )

    # Dynamic decoding
    outputs_infer, _, _ = seq2seq.dynamic_decode(
        my_decoder_infer,
        maximum_iterations=hparams.max_inference_len,
        output_time_major=False,
        swap_memory=True,
        scope=decoder_scope)

    if model.mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
      logits_infer = tf.no_op()
      sample_id_infer = outputs_infer.predicted_ids
    else:
      logits_infer = outputs_infer.rnn_output
      sample_id_infer = outputs_infer.sample_id

  return logits_train, logits_infer, sample_id_train, sample_id_infer


def _build_action_decoder_cell(model, hparams, encoder_state, base_gpu):
  """decoder cell constructor for action states."""
  num_residual_layers = hparams.num_residual_layers
  cell = model_helper.create_rnn_cell(
      num_units=hparams.num_units,
      num_layers=1,
      num_residual_layers=num_residual_layers,
      dropout=hparams.dropout,
      num_gpus=hparams.num_gpus,
      mode=model.mode,
      single_cell_fn=model.single_cell_fn,
      base_gpu=base_gpu)

  # For beam search, we need to replicate encoder infos beam_width times
  if model.mode == tf.estimator.ModeKeys.PREDICT and hparams.beam_width > 0:
    decoder_initial_state = seq2seq.tile_batch(
        encoder_state[-1], multiplier=hparams.beam_width)
  else:
    decoder_initial_state = encoder_state[-1]

  return cell, decoder_initial_state


def _build_decoder_action(model, dialogue_state, hparams, start_token,
                          end_token, output_layer):
  """build the decoder for action states."""

  iterator = model.iterator

  start_token_id = tf.cast(
      model.vocab_table.lookup(tf.constant(start_token)), tf.int32)
  end_token_id = tf.cast(
      model.vocab_table.lookup(tf.constant(end_token)), tf.int32)

  start_tokens = tf.fill([model.batch_size], start_token_id)
  end_token = end_token_id

  # kb is not used again
  ## Decoder.
  with tf.variable_scope("action_decoder") as decoder_scope:
    # we initialize the cell with the last layer of the last hidden state
    cell, decoder_initial_state = _build_action_decoder_cell(
        model, hparams, dialogue_state, model.global_gpu_num)
    model.global_gpu_num += 1
    ## Train or eval
    # situation one, for train, eval, mutable train
    # decoder_emp_inp: [max_time, batch_size, num_units]
    decoder_emb_inp = tf.nn.embedding_lookup(model.embedding_decoder,
                                             iterator.action)

    # Helper
    helper_train = seq2seq.TrainingHelper(
        decoder_emb_inp, iterator.action_len, time_major=False)

    # Decoder
    my_decoder_train = seq2seq.BasicDecoder(
        cell, helper_train, decoder_initial_state, output_layer)

    # Dynamic decoding
    outputs_train, _, _ = seq2seq.dynamic_decode(
        my_decoder_train,
        output_time_major=False,
        swap_memory=True,
        scope=decoder_scope)

    sample_id_train = outputs_train.sample_id
    logits_train = outputs_train.rnn_output
    # inference

    beam_width = hparams.beam_width
    length_penalty_weight = hparams.length_penalty_weight

    if model.mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
      my_decoder_infer = seq2seq.BeamSearchDecoder(
          cell=cell,
          embedding=model.embedding_decoder,
          start_tokens=start_tokens,
          end_token=end_token,
          initial_state=decoder_initial_state,
          beam_width=beam_width,
          output_layer=output_layer,
          length_penalty_weight=length_penalty_weight)
    else:
      # Helper
      if model.mode in dialogue_utils.self_play_modes:
        helper_infer = seq2seq.SampleEmbeddingHelper(
            model.embedding_decoder, start_tokens, end_token)
      else:
        helper_infer = seq2seq.GreedyEmbeddingHelper(
            model.embedding_decoder, start_tokens, end_token)

      # Decoder
      my_decoder_infer = seq2seq.BasicDecoder(
          cell,
          helper_infer,
          decoder_initial_state,
          output_layer=output_layer  # applied per timestep
      )

    # Dynamic decoding
    outputs_infer, _, _ = seq2seq.dynamic_decode(
        my_decoder_infer,
        maximum_iterations=hparams.len_action,
        output_time_major=False,
        swap_memory=True,
        scope=decoder_scope)

    if model.mode == tf.estimator.ModeKeys.PREDICT and beam_width > 0:
      logits_infer = tf.no_op()
      sample_id_infer = outputs_infer.predicted_ids
    else:
      logits_infer = outputs_infer.rnn_output
      sample_id_infer = outputs_infer.sample_id

  return logits_train, logits_infer, sample_id_train, sample_id_infer


def _build_encoder_simple(model, intent, intent_length, num_units):
  """Build an encoder for intent."""
  with tf.variable_scope("encoder") as scope:
    dtype = scope.dtype
    # Look up embedding, emp_inp: [max_time, batch_size, num_units]
    encoder_emb_inp = tf.nn.embedding_lookup(model.embedding_encoder, intent)

    cell = model_helper._single_cell(
        num_units,
        model.hparams.dropout,
        model.mode,
        residual_connection=False,
        device_str=model_helper.get_device_str(model.global_gpu_num,
                                               model.hparams.num_gpus))
    model.global_gpu_num += 1

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        cell,
        encoder_emb_inp,
        dtype=dtype,
        sequence_length=intent_length,
        time_major=False,
        swap_memory=True)

  return encoder_outputs, encoder_state, encoder_emb_inp


def _build_encoder_hierarchial(model, data_source, num_units):
  """Build an encoder for kb."""

  source = data_source  # bs*num_entry, 13

  with tf.variable_scope("encoder") as scope:
    dtype = scope.dtype
    # Look up embedding, emp_inp: [max_time, batch_size, num_units]
    encoder_emb_inp = tf.nn.embedding_lookup(model.embedding_encoder, source)

    # Encoder_outpus: [max_time, batch_size, num_units]
    cell_0 = model_helper._single_cell(
        num_units,
        model.hparams.dropout,
        model.mode,
        residual_connection=False,
        device_str=model_helper.get_device_str(model.global_gpu_num,
                                               model.hparams.num_gpus))
    model.global_gpu_num += 1
    with tf.variable_scope("hierarchial_rnn_1") as scope:
      _, encoder_final_states_0 = tf.nn.dynamic_rnn(
          cell_0,
          encoder_emb_inp,
          dtype=dtype,
          time_major=False,
          swap_memory=True)
    encoder_final_states_0 = tf.reshape(encoder_final_states_0,
                                        [model.batch_size, -1, num_units])
    cell_1 = model_helper._single_cell(
        num_units,
        model.hparams.dropout,
        model.mode,
        residual_connection=False,
        device_str=model_helper.get_device_str(model.global_gpu_num,
                                               model.hparams.num_gpus))
    model.global_gpu_num += 1
    with tf.variable_scope("hierarchial_rnn_2") as scope:
      encoder_outputs_1, encoder_state_1 = tf.nn.dynamic_rnn(
          cell_1,
          encoder_final_states_0,
          dtype=dtype,
          time_major=False,
          swap_memory=True)
  return encoder_outputs_1, encoder_state_1, encoder_emb_inp


def _build_value_network(model,
                         encoder_emb_inp,
                         action_emb_inp,
                         aux_hidden_state,
                         transform_layer1,
                         transform_layer2,
                         hparams,
                         has_emb_input=False):
  """build value network."""
  encoder_emb_inp = tf.stop_gradient(encoder_emb_inp)
  action_emb_inp = tf.stop_gradient(action_emb_inp)
  aux_hidden_state = tf.stop_gradient(aux_hidden_state)  # add stop gradient to
  # 1. do projection
  projected1 = transform_layer1(encoder_emb_inp)
  projected1a = transform_layer1(action_emb_inp)
  projected2 = transform_layer2(aux_hidden_state)
  # 2. tile the aux one
  num_time = tf.shape(encoder_emb_inp)[1]
  num_actions = tf.shape(projected1a)[1]
  projected2 = tf.reshape(projected2,
                          [model.batch_size, 1, hparams.unit_value_network])
  projected2 = tf.tile(projected2, [1, num_time, 1])
  with tf.variable_scope("value_network"):
    dialogue_value_function = tf.multiply(projected1,
                                          projected2)  # bs,time, num_units
    dialogue_value_function = tf.reduce_sum(dialogue_value_function,
                                            -1)  # bs,time
    if has_emb_input:
      # bs,action tokens, num_units, projected2 has the same content
      action_value_function = tf.multiply(projected1a,
                                          projected2[:, 0:num_actions, :])
      action_value_function = tf.reduce_sum(action_value_function,
                                            -1)  # bs,time
    else:
      action_value_function = None

  return dialogue_value_function, action_value_function


def build_graph(model, hparams, scope=None):
  """build the computation graph."""
  utils.print_out("# creating %s graph ..." % model.mode)
  dtype = tf.float32
  num_layers = hparams.num_layers
  num_gpus = hparams.num_gpus

  with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
    # Encoder
    # Look up embedding, emp_inp: [max_time, batch_size, num_units]
    with tf.variable_scope("encoder_emb_inp"):
      encoder_emb_inp = tf.nn.embedding_lookup(model.embedding_encoder,
                                               model.iterator.source)
      action_emb_inp = tf.nn.embedding_lookup(model.embedding_encoder,
                                              model.iterator.action)
    with tf.variable_scope("encoder1_intent"):
      res = _build_encoder_simple(
          model,
          model.iterator.intent,
          model.iterator.intent_len,
          num_units=hparams.encoder_intent_unit)
      _, encoder_state1_aux, _ = res
    with tf.variable_scope("encoder2_kb"):
      res = _build_encoder_hierarchial(
          model, model.iterator.kb, num_units=hparams.encoder_kb_unit)
      _, encoder_state2_aux, _ = res

    with tf.variable_scope("encoder1"):
      model.encoder_input_projection1 = tf.layers.Dense(
          hparams.num_units, use_bias=False, name="encoder_1_input_projection")
      tiled_encoder_state1_aux = tf.reshape(
          encoder_state1_aux,
          [model.batch_size, 1, hparams.encoder_intent_unit])
      time_step = tf.shape(encoder_emb_inp)[1]
      tiled_encoder_state1_aux = tf.tile(tiled_encoder_state1_aux,
                                         [1, time_step, 1])
      concat1 = tf.concat([encoder_emb_inp, tiled_encoder_state1_aux],
                          2)  # emb_intnt+num_unites
      encoder1_input = model.encoder_input_projection1(concat1)
      encoder_outputs1, encoder_state1 = _build_encoder(
          model, encoder1_input, hparams)  # 1= customer, 2= agent

    with tf.variable_scope("encoder2"):
      model.encoder_input_projection2 = tf.layers.Dense(
          hparams.num_units, use_bias=False, name="encoder_2_input_projection")
      tiled_encoder_state2_aux = tf.reshape(
          encoder_state2_aux, [model.batch_size, 1, hparams.encoder_kb_unit])
      time_step = tf.shape(encoder_emb_inp)[1]
      tiled_encoder_state2_aux = tf.tile(tiled_encoder_state2_aux,
                                         [1, time_step, 1])
      concat2 = tf.concat([encoder_emb_inp, tiled_encoder_state2_aux],
                          2)  # emb_intnt+num_unites
      encoder2_input = model.encoder_input_projection2(concat2)
      encoder_outputs2, encoder_state2 = _build_encoder(model, encoder2_input,
                                                        hparams)

    ## Decoder
    with tf.variable_scope("decoder1"):
      res = _build_decoder(model, encoder_outputs1, encoder_state1, hparams,
                           vocab_utils.start_of_turn1,
                           vocab_utils.start_of_turn2, model.output_layer1,
                           encoder_state1_aux)
      logits_trian1, _, sample_id_train1, sample_id_infer1 = res

    with tf.variable_scope("decoder2"):
      res = _build_decoder(model, encoder_outputs2, encoder_state2, hparams,
                           vocab_utils.start_of_turn2,
                           vocab_utils.start_of_turn1, model.output_layer2,
                           encoder_state2_aux)
      logits_trian2, _, sample_id_train2, sample_id_infer2 = res

    with tf.variable_scope("decoder_action"):
      res = _build_decoder_action(
          model,
          encoder_state2,
          hparams,
          hparams.t1.encode(),  # dialogue ends with t2, action starts with t1
          hparams.t2.encode(),
          model.output_layer_action)
      logits_trian3, _, sample_id_train3, sample_id_infer3 = res

    with tf.variable_scope("value_network1"):
      res = _build_value_network(model, encoder_emb_inp, action_emb_inp,
                                 encoder_state1_aux, model.vn_project11,
                                 model.vn_project12, hparams)
      dialogue1_val, _ = res
    with tf.variable_scope("value_network2"):
      res = _build_value_network(model, encoder_emb_inp, action_emb_inp,
                                 encoder_state2_aux, model.vn_project21,
                                 model.vn_project22, hparams, True)
      dialogue2_val, action_val = res

      model.logits_trian1 = logits_trian1
      model.logits_trian2 = logits_trian2
      model.dialogue1_val = dialogue1_val
      model.dialogue2_val = dialogue2_val

    if model.mode in [
        tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL,
        dialogue_utils.mode_self_play_mutable
    ]:
      with tf.device(model_helper.get_device_str(num_layers - 1, num_gpus)):
        sl_loss, sl_loss_arr = _compute_loss(model, logits_trian1,
                                             logits_trian2, logits_trian3)

      with tf.device(model_helper.get_device_str(num_layers - 1, num_gpus)):
        rl_loss_arr = _compute_loss_selfplay(
            model, logits_trian1, logits_trian2, logits_trian3, dialogue1_val,
            dialogue2_val, action_val)

    elif model.mode == tf.estimator.ModeKeys.PREDICT or model.mode == dialogue_utils.mode_self_play_immutable:
      sl_loss, sl_loss_arr, rl_loss_arr = None, None, None
    else:
      raise ValueError("mode not known")

    sample_id_arr_train = [sample_id_train1, sample_id_train2, sample_id_train3]
    sample_id_arr_infer = [sample_id_infer1, sample_id_infer2, sample_id_infer3]

    return sl_loss, sl_loss_arr, rl_loss_arr, sample_id_arr_train, sample_id_arr_infer


def _compute_loss(model, logits1, logits2, logits3):
  """Compute optimization loss for supervised learning."""

  target_output = model.iterator.target
  crossent1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_output, logits=logits1)  # calculate excludes the last one
  crossent2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_output, logits=logits2)  # calculate excludes the last one
  crossent3 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=model.iterator.action,
      logits=logits3)  # calculate excludes the last one

  target_weights1 = tf.cast(model.iterator.mask1, dtype=logits1.dtype)
  target_weights2 = tf.cast(model.iterator.mask2, dtype=logits2.dtype)

  loss1 = tf.reduce_sum(tf.multiply(crossent1, target_weights1)) / tf.cast(
      model.batch_size, tf.float32)
  loss2 = tf.reduce_sum(tf.multiply(crossent2, target_weights2)) / tf.cast(
      model.batch_size, tf.float32)
  loss3 = tf.reduce_sum(crossent3) / tf.cast(model.batch_size, tf.float32)

  probs = tf.nn.softmax(logits3, -1)  # bs, len_action, vocab
  predictions = tf.argmax(probs, axis=2)  # bs, len_action, 1

  predictions = tf.reshape(predictions,
                           [tf.shape(predictions)[0], model.hparams.len_action])
  action = model.iterator.action
  len_action = model.hparams.len_action
  name_length = len_action - 2
  acc1 = tf.reduce_mean(
      tf.cast(
          tf.equal(
              tf.cast(action[:, 0:name_length], tf.int32),
              tf.cast(predictions[:, 0:name_length], tf.int32)), tf.float32))
  acc2 = tf.reduce_mean(
      tf.cast(
          tf.equal(
              tf.cast(action[:, name_length], tf.int32),
              tf.cast(predictions[:, name_length], tf.int32)), tf.float32))
  acc3 = tf.reduce_mean(
      tf.cast(
          tf.equal(
              tf.cast(action[:, name_length + 1], tf.int32),
              tf.cast(predictions[:, name_length + 1], tf.int32)), tf.float32))
  return loss1 + loss2 + loss3, [
      loss1,  # they will be normalized against batch size later
      loss2,
      loss3,
      acc1,
      acc2,
      acc3
  ]


def _compute_loss_selfplay(model, logits1, logits2, logits3, value_da1,
                           value_da2, value_ac2):  ###
  """loss function for selfplay."""
  # 0.common variables

  target_weights1 = tf.cast(model.iterator.mask1, dtype=tf.float32)
  target_weights2 = tf.cast(model.iterator.mask2, dtype=tf.float32)

  # 1. value network
  # [bs,padding_size], here is aligned with padding and holes
  reward_dialogue = model.iterator.reward_diag
  reward_action = model.iterator.reward_action  # [bs, len_action]
  action_size = tf.shape(reward_action)[1]

  concat_reward_dialogue = tf.concat([reward_dialogue, reward_action], axis=-1)
  concat_value_da2 = tf.concat([value_da2, value_ac2], axis=-1)
  concat_target_weights2 = tf.concat(
      [
          target_weights2,
          tf.ones(
              [model.batch_size, model.hparams.len_action],
              dtype=tf.float32,
              name=None)
      ],
      axis=-1)

  final_value_loss_1 = tf.losses.mean_squared_error(
      reward_dialogue,
      value_da1,
      weights=target_weights1,
      scope=None,
      loss_collection=tf.GraphKeys.LOSSES,
      reduction=tf.losses.Reduction.SUM)

  final_value_loss_2 = tf.losses.mean_squared_error(
      concat_reward_dialogue,
      concat_value_da2,
      weights=concat_target_weights2,
      scope=None,
      loss_collection=tf.GraphKeys.LOSSES,
      reduction=tf.losses.Reduction.SUM)
  # 2. policy network
  tar = model.iterator.target

  # logits1=logits2= [bs, padding_size, vocab]
  # tar = [bs, padding_size, 1]
  # logits3=[bs, 3, vocab]
  # iterator.action = [bs, 3, 1]
  crossent1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tar, logits=logits1)  # calculate excludes the last one
  crossent2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tar, logits=logits2)  # calculate excludes the last one
  crossent3 = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=model.iterator.action,
      logits=logits3)  # calculate excludes the last one

  prob1_masked = tf.multiply(crossent1, target_weights1)
  prob2_masked = tf.multiply(crossent2, target_weights2)

  # prob3 = tf.reduce_sum(crossent3, -1)  # bs, 3
  prob3 = tf.reshape(crossent3,
                     [tf.shape(crossent3)[0], model.hparams.len_action])

  if model.hparams.self_play_loss_method == 1:
    eq11 = model.iterator.reward_diag - tf.stop_gradient(value_da1)
    eq12 = model.iterator.reward_diag - tf.stop_gradient(value_da2)
    eq2 = model.iterator.reward_action - tf.stop_gradient(value_ac2)
  elif model.hparams.self_play_loss_method == 2:
    eq11 = model.iterator.reward_diag - tf.tile(
        tf.reshape(tf.reduce_mean(model.iterator.reward_diag, 0), [1, -1]),
        [model.batch_size, 1])
    eq12 = model.iterator.reward_diag - tf.tile(
        tf.reshape(tf.reduce_mean(model.iterator.reward_diag, 0), [1, -1]),
        [model.batch_size, 1])
    eq2 = model.iterator.reward_action - tf.tile(
        tf.reshape(tf.reduce_mean(model.iterator.reward_action, 0), [1, -1]),
        [model.batch_size, 1])
  else:
    raise ValueError("invalid case in compute loss selfplay")

  policy_gradient1 = tf.multiply(prob1_masked,
                                 eq11)  # policy gradient for utt speaker 1
  policy_gradient2 = tf.multiply(prob2_masked,
                                 eq12)  # policy gradient for utt speaker 2
  policy_gradient3 = tf.multiply(prob3,
                                 eq2)  # policy gradient for action speaker 2
  policy_gradient3 = tf.reduce_sum(policy_gradient3, -1)

  gain1_raw = -1 * tf.reduce_sum(policy_gradient1)
  gain2_raw = -1 * tf.reduce_sum(policy_gradient2)
  gain3_raw = -1 * tf.reduce_sum(policy_gradient3)
  final_policy_gain1 = gain1_raw / tf.reduce_sum(target_weights1)
  final_policy_gain2 = gain2_raw + gain3_raw
  gain2_denominator = tf.reduce_sum(target_weights2) + tf.cast(
      model.batch_size * action_size, tf.float32)
  final_policy_gain2 = final_policy_gain2 / gain2_denominator

  return final_value_loss_1, final_value_loss_2, final_policy_gain1, final_policy_gain2, eq11, eq12, eq2
