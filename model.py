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

"""Main module for the dialogue generation model."""
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import layers
import model_helper
from build_graph import build_graph
from utils import dialogue_utils
from utils import iterator_utils
from utils import misc_utils as utils


class Model(object):
  """class for dialogue model."""

  def __init__(self,
               hparams,
               mode,
               iterator,
               handle,
               vocab_table,
               reverse_vocab_table=None,
               scope=None,
               extra_args=None):
    assert isinstance(iterator, iterator_utils.BatchedInput)
    self.iterator = iterator
    self.handle = handle
    self.mode = mode
    self.vocab_table = vocab_table
    self.vocab_size = hparams.vocab_size
    self.num_layers = hparams.num_layers
    self.num_gpus = hparams.num_gpus
    self.hparams = hparams
    self.single_cell_fn = None
    self.global_gpu_num = 0
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Initializer
    initializer = model_helper.get_initializer(
        hparams.init_op, hparams.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    self.init_embeddings(hparams, scope)
    self.batch_size = tf.shape(self.iterator.source)[0]

    # Projection
    with tf.variable_scope(scope or "build_network"):
      with tf.variable_scope("decoder/output_projection"):
        self.output_layer1 = layers.Dense(
            hparams.vocab_size, use_bias=False, name="output_projection_1")
        self.output_layer2 = layers.Dense(
            hparams.vocab_size, use_bias=False, name="output_projection_2")
        self.output_layer_action = layers.Dense(
            hparams.vocab_size, use_bias=False, name="output_projection_action")
        self.vn_project11 = layers.Dense(
            hparams.unit_value_network, use_bias=False, name="vn_project_11")
        self.vn_project12 = layers.Dense(
            hparams.unit_value_network, use_bias=False, name="vn_project_12")
        self.vn_project21 = layers.Dense(
            hparams.unit_value_network, use_bias=False, name="vn_project_21")
        self.vn_project22 = layers.Dense(
            hparams.unit_value_network, use_bias=False, name="vn_project_22")

    ## Train graph
    sl_loss, sl_loss_arr, rl_loss_arr, sample_id_arr_train, sample_id_arr_infer = build_graph(
        self, hparams, scope=scope)

    if self.mode == tf.estimator.ModeKeys.TRAIN:
      self.train_loss = sl_loss
      self.all_train_loss = sl_loss_arr
      self.word_count = tf.reduce_sum(self.iterator.dialogue_len)
      self.sample_ids_arr = sample_id_arr_train
      self.sample_words_arr1 = []
      self.sample_words_arr2 = []
      source = self.iterator.source
      for i in range(len(self.sample_ids_arr)):
        element_infer = self.sample_ids_arr[i]
        element_src = source[0]
        # element_src=0
        src = reverse_vocab_table.lookup(tf.cast(element_src, tf.int64))
        infer = reverse_vocab_table.lookup(
            tf.cast(element_infer, tf.int64)
        )[0]  # src can only get the first one so I only get the first inference
        if i == 0:
          self.sample_words_arr1.append((tf.constant(i), src, infer))
        elif i == 1:
          self.sample_words_arr2.append((tf.constant(i), src, infer))
      self.vl1, self.vl2, self.pl1, self.pl2, self.eq11, self.eq12, self.eq2 = rl_loss_arr  # reinforcement updates

    elif self.mode == tf.estimator.ModeKeys.EVAL:
      self.eval_loss = sl_loss
      self.all_eval_loss = sl_loss_arr

    elif self.mode == tf.estimator.ModeKeys.PREDICT:
      self.sample_ids_arr = sample_id_arr_infer
      self.sample_words_arr = []
      self.source = reverse_vocab_table.lookup(tf.cast(iterator.source, tf.int64))
      for element in self.sample_ids_arr:
        self.sample_words_arr.append(
            reverse_vocab_table.lookup(tf.cast(element, tf.int64)))
    elif self.mode in dialogue_utils.self_play_modes:
      #### self play
      self.train_loss = sl_loss
      self.all_train_loss = sl_loss_arr
      self.selfplay_agent_1_utt = reverse_vocab_table.lookup(
          tf.cast(sample_id_arr_infer[0], tf.int64))
      self.selfplay_agent_2_utt = reverse_vocab_table.lookup(
          tf.cast(sample_id_arr_infer[1], tf.int64))
      self.selfplay_action = reverse_vocab_table.lookup(
          tf.cast(sample_id_arr_infer[2], tf.int64))
      if self.mode == dialogue_utils.mode_self_play_mutable:
        self.vl1, self.vl2, self.pl1, self.pl2, self.eq11, self.eq12, self.eq2 = rl_loss_arr  # reinforcement updates

    if self.mode != tf.estimator.ModeKeys.PREDICT:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(self.iterator.dialogue_len)

    ## Learning rate
    warmup_steps = hparams.learning_rate_warmup_steps
    warmup_factor = hparams.learning_rate_warmup_factor
    print("  start_decay_step=%d, learning_rate=%g, decay_steps %d, "
          "decay_factor %g, learning_rate_warmup_steps=%d, "
          "learning_rate_warmup_factor=%g, starting_learning_rate=%g" %
          (hparams.start_decay_step, hparams.learning_rate, hparams.decay_steps,
           hparams.decay_factor, warmup_steps, warmup_factor,
           (hparams.learning_rate * warmup_factor**warmup_steps)))
    self.global_step = tf.Variable(0, trainable=False)

    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
    if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == dialogue_utils.mode_self_play_mutable:
      self.learning_rate = tf.constant(hparams.learning_rate)

      inv_decay = warmup_factor**(tf.cast(warmup_steps - self.global_step, tf.float32))
      self.learning_rate = tf.cond(
          self.global_step < hparams.learning_rate_warmup_steps,
          lambda: inv_decay * self.learning_rate,
          lambda: self.learning_rate,
          name="learning_rate_decay_warump_cond")

      if hparams.optimizer == "sgd":
        self.learning_rate = tf.cond(
            self.global_step < hparams.start_decay_step,
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - hparams.start_decay_step),
                hparams.decay_steps,
                hparams.decay_factor,
                staircase=True),
            name="sgd_learning_rate_supervised")
        opt = tf.train.GradientDescentOptimizer(
            self.learning_rate, name="SGD_supervised")
        tf.summary.scalar("lr", self.learning_rate)
      elif hparams.optimizer == "adam":
        assert float(
            hparams.learning_rate
        ) <= 0.001, "! High Adam learning rate %g" % hparams.learning_rate
        opt = tf.train.AdamOptimizer(self.learning_rate, name="Adam_supervised")

      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops,
          name="gradients_adam")

      clipped_gradients, gradient_norm_summary = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)

      self.update = opt.apply_gradients(
          list(zip(clipped_gradients, params)),
          global_step=self.global_step,
          name="adam_apply_gradients")

      # Summary
      self.train_summary = tf.summary.merge([
          tf.summary.scalar("lr", self.learning_rate),
          tf.summary.scalar("train_loss", self.train_loss),
      ] + gradient_norm_summary)

    # second part of the learning rate
    if self.mode == tf.estimator.ModeKeys.TRAIN or self.mode == dialogue_utils.mode_self_play_mutable:
      self.learning_rate2 = tf.constant(hparams.learning_rate2)
      self.learning_rate3 = tf.constant(hparams.learning_rate3)
      if hparams.optimizer == "sgd":
        self.learning_rate2 = tf.cond(
            self.global_step < hparams.start_decay_step,
            lambda: self.learning_rate2,
            lambda: tf.train.exponential_decay(
                self.learning_rate2,
                (self.global_step - hparams.start_decay_step),
                hparams.decay_steps,
                hparams.decay_factor,
                staircase=True),
            name="sgd_learning_rate_supervised2")
        self.learning_rate3 = tf.cond(
            self.global_step < hparams.start_decay_step,
            lambda: self.learning_rate3,
            lambda: tf.train.exponential_decay(
                self.learning_rate3,
                (self.global_step - hparams.start_decay_step),
                hparams.decay_steps,
                hparams.decay_factor,
                staircase=True),
            name="sgd_learning_rate_supervised3")
        tf.summary.scalar("self_play_lr", self.learning_rate)
      elif hparams.optimizer == "adam":
        assert float(
            hparams.learning_rate2
        ) <= 0.001, "! High Adam learning rate2 %g" % hparams.learning_rate2
        assert float(
            hparams.learning_rate3
        ) <= 0.001, "! High Adam learning rate3 %g" % hparams.learning_rate3

      # params=[]

      print("params=")
      for element in params:
        print(element.name)
      val1_params = self.patial_params(params,
                                       ["dynamic_seq2seq/value_network1"])
      val2_params = self.patial_params(params,
                                       ["dynamic_seq2seq/value_network2"])
      embedding_params = self.patial_params(params, ["embeddings"])
      main_dec_enc_params1 = self.patial_params(
          params, ["dynamic_seq2seq/encoder1/", "dynamic_seq2seq/decoder1/"])
      main_dec_enc_params2 = self.patial_params(
          params, ["dynamic_seq2seq/encoder2/", "dynamic_seq2seq/decoder2/"])
      action_params = self.patial_params(params,
                                         ["dynamic_seq2seq/decoder_action"])
      encoder_kb_params = self.patial_params(params,
                                             ["dynamic_seq2seq/encoder2_kb"])
      encoder_intent_params = self.patial_params(
          params, ["dynamic_seq2seq/encoder1_intent"])
      print("val1_params", "\n".join([a.name for a in val1_params]))
      print("val2_params", "\n".join([a.name for a in val2_params]))
      print("embedding_params", "\n".join(
          [a.name for a in embedding_params]))
      print("main_dec_enc_params1", "\n".join(
          [a.name for a in main_dec_enc_params1]))
      print("main_dec_enc_params2", "\n".join(
          [a.name for a in main_dec_enc_params2]))
      print("action_params", "\n".join([a.name for a in action_params]))
      print("encoder_kb_params", "\n".join(
          [a.name for a in encoder_kb_params]))
      print("encoder_intent_params", "\n".join(
          [a.name for a in encoder_intent_params]))
      self.optimizer_vl1, self.v1_sum = self.generate_optimizer(
          self.vl1, params, "vl1", self.learning_rate2,
          self.hparams.max_gradient_norm2)
      self.optimizer_vl2, self.v2_sum = self.generate_optimizer(
          self.vl2, params, "vl2", self.learning_rate2,
          self.hparams.max_gradient_norm2)
      if hparams.self_play_variable_method == 0:
        rl_param1, rl_param2 = encoder_intent_params, encoder_kb_params + action_params
      elif hparams.self_play_variable_method == 1:
        rl_param1, rl_param2 = main_dec_enc_params1, main_dec_enc_params2
      elif hparams.self_play_variable_method == 2:
        rl_param1, rl_param2 = main_dec_enc_params1 + encoder_intent_params, main_dec_enc_params2 + encoder_kb_params + action_params
      elif hparams.self_play_variable_method == 3:
        rl_param1, rl_param2 = [
            main_dec_enc_params1[0]
        ] + encoder_intent_params, [main_dec_enc_params2[0]] + encoder_kb_params
      elif hparams.self_play_variable_method == 4:
        rl_param1, rl_param2 = [main_dec_enc_params1[0]], [
            main_dec_enc_params2[0]
        ]
      elif hparams.self_play_variable_method == 5:
        rl_param1, rl_param2 = params, params
      self.optimizer_pl1, self.p1_sum = self.generate_optimizer(
          self.pl1, params, "pl1", self.learning_rate3,
          self.hparams.max_gradient_norm3)
      self.optimizer_pl2, self.p2_sum = self.generate_optimizer(
          self.pl2, params, "pl2", self.learning_rate3,
          self.hparams.max_gradient_norm3)
      print("self.learning", self.learning_rate, self.learning_rate2,
            self.learning_rate3)
      ################################
      ### supervised learning######'
      ###########################
    # Saver
    self.saver = tf.train.Saver(tf.global_variables())

    # Print trainable variables
    utils.print_out("# Trainable variables")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

  def generate_optimizer(self, loss, params, name, learning_rate,
                         max_gradient_norm):
    """generates optimizer."""
    if self.hparams.optimizer == "sgd":
      opt = tf.train.GradientDescentOptimizer(
          learning_rate, name="SGD_self_play_" + name)
    else:
      opt = tf.train.AdamOptimizer(learning_rate, name="ADAM_self_play_" + name)

    gradients = tf.gradients(
        loss,
        params,
        colocate_gradients_with_ops=self.hparams.colocate_gradients_with_ops,
        name="gradients_" + name)

    clipped_gradients, gradient_norm_summary = model_helper.gradient_clip(
        gradients, max_gradient_norm=max_gradient_norm)

    update = opt.apply_gradients(
        list(zip(clipped_gradients, params)), global_step=self.global_step, name=name)

    return update, gradient_norm_summary

  def patial_params(self, params, prefix_set):
    """get the set of parameters that belong to a prefix_set."""
    new_params = []
    for p in params:
      for prefix in prefix_set:
        if p.name.startswith(prefix):
          new_params.append(p)
          break
    return new_params

  def init_embeddings(self, hparams, scope):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            vocab_size=self.vocab_size,
            embed_size=hparams.num_units,
            num_partitions=hparams.num_embeddings_partitions,
            scope=scope,
        ))

  def train(self, sess, iterator_handle):
    """supervised learning train step."""
    assert self.mode == tf.estimator.ModeKeys.TRAIN
    summaries = {
        "train_dialogue_loss1": self.all_train_loss[0],
        "train_dialogue_loss2": self.all_train_loss[1],
        "train_action_loss3": self.all_train_loss[2],
        "train_action_acc1": self.all_train_loss[3],
        "train_action_acc2": self.all_train_loss[4],
        "train_action_acc3": self.all_train_loss[5]
    }
    res = sess.run(
        [
            self.update, self.train_loss, summaries, self.predict_count,
            self.train_summary, self.global_step, self.word_count,
            self.batch_size, self.iterator.source, self.iterator.target,
            self.sample_words_arr1, self.sample_words_arr2, self.iterator.mask1,
            self.iterator.mask2
        ],
        feed_dict={self.handle: iterator_handle})

    return res

  def self_play_train(self, sess, iterator_handle):
    """supervised training step during self play."""
    assert self.mode == dialogue_utils.mode_self_play_mutable
    summaries = {
        "SL_dialogue_loss1": self.all_train_loss[0],
        "SL_dialogue_loss2": self.all_train_loss[1],
        "SL_action_loss3": self.all_train_loss[2],
        "SL_train_loss": self.train_loss,
    }
    res = sess.run(
        [self.update, self.global_step, summaries],
        feed_dict={self.handle: iterator_handle})
    return res

  def eval(self, sess, iterator_handle):
    """eval step."""
    assert self.mode == tf.estimator.ModeKeys.EVAL
    summaries = {
        "eval_dialogue_loss1": self.all_eval_loss[0],
        "eval_dialogue_loss2": self.all_eval_loss[1],
        "eval_action_loss3": self.all_eval_loss[2],
        "eval_action_acc1": self.all_eval_loss[3],
        "eval_action_acc2": self.all_eval_loss[4],
        "eval_action_acc3": self.all_eval_loss[5]
    }
    return sess.run(
        [self.eval_loss, summaries, self.predict_count, self.batch_size],
        feed_dict={self.handle: iterator_handle})

  def self_play(self, sess, speaker, self_play_handle):
    """selfplay dialogue generation step."""
    assert self.mode == dialogue_utils.mode_self_play_mutable
    if speaker == 0:
      execution = [
          self.optimizer_vl1, self.optimizer_pl1, self.global_step, self.eq11,
          self.logits_trian1, self.dialogue2_val, {
              "RL_value_loss1": self.vl1,
              "RL_neg_policy_gain1": self.pl1,
              "RL_dialogue_loss1": self.all_train_loss[0],
              "RL_dialogue_loss2": self.all_train_loss[1],
              "RL_action_loss3": self.all_train_loss[2],
              "RL_train_loss": self.train_loss,
          }
      ]
    else:
      execution = [
          self.optimizer_vl2, self.optimizer_pl2, self.global_step, self.eq12,
          self.logits_trian2, {
              "RL_value_loss2": self.vl2,
              "RL_neg_policy_gain2": self.pl2,
              "RL_dialogue_loss1": self.all_train_loss[0],
              "RL_dialogue_loss2": self.all_train_loss[1],
              "RL_action_loss3": self.all_train_loss[2],
              "RL_train_loss": self.train_loss,
          }
      ]
    res = sess.run(execution, feed_dict={self.handle: self_play_handle})

    return res

  def generate_utterance(self, sess, speaker, iterator_handle):
    """generate utterance for sample decoder."""
    sample_words_tf = self.sample_words_arr[speaker]  # speaker is either 0 or 1

    sample_words_actual, source = sess.run(
        [sample_words_tf, self.source],
        feed_dict={self.handle: iterator_handle})

    sample_words_actual[0] = dialogue_utils.extract_best_beam_single(
        sample_words_actual[0])
    # make sure outputs is of shape [batch_size, time]
    return sample_words_actual, None, source

  def generate_self_play_utterance(self, sess, iterator_handle):
    """generate utterance for selfplay."""
    response = sess.run(
        [
            self.selfplay_agent_1_utt, self.selfplay_agent_2_utt,
            self.selfplay_action
        ],
        feed_dict={self.handle: iterator_handle})
    new_response = dialogue_utils.extract_best_beam_response(response)
    return new_response

  def generate_infer_utterance(self, sess, infer_iterator_handle):
    """generate utterance for inference."""
    response = sess.run(
        self.sample_words_arr, feed_dict={self.handle: infer_iterator_handle})

    new_response = dialogue_utils.extract_best_beam_response(response)
    return new_response
