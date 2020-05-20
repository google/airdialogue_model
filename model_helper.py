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

"""Utility functions for building models."""


import collections
import logging
import re
import time
import tensorflow.compat.v1 as tf
from rnn_decoder.multi_rnn import MultiRNNCell
from utils import dialogue_utils
from utils import iterator_utils
from utils import misc_utils as utils
from utils import vocab_utils


def get_initializer(init_op, seed=None, init_weight=None):
  """Create an initializer. init_weight is only for uniform."""
  if init_op == "uniform":
    assert init_weight
    return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
  elif init_op == "glorot_normal":
    return tf.keras.initializers.glorot_normal(seed=seed)
  elif init_op == "glorot_uniform":
    return tf.keras.initializers.glorot_uniform(seed=seed)
  else:
    raise ValueError("Unknown init_op %s" % init_op)


def get_device_str(device_id, num_gpus):
  """Return a device string for multi-GPU setup."""
  if num_gpus == 0:
    return "/cpu:0"
  device_str_output = "/gpu:%d" % (device_id % num_gpus)
  return device_str_output


class ExtraArgs(
    collections.namedtuple(
        "ExtraArgs",
        ("single_cell_fn", "model_device_fn", "attention_mechanism_fn"))):
  pass


class TrainModel(
    collections.namedtuple(
        "TrainModel",
        ("graph", "model", "placeholder_iterator", "placeholder_handle",
         "train_iterator", "skip_count_placeholder"))):
  pass


class EvalModel(
    collections.namedtuple(
        "EvalModel",
        ("graph", "model", "placeholder_iterator", "placeholder_handle",
         "eval_iterator", "data_file_placeholder", "kb_file_placeholder"))):
  pass


class InferModel(
    collections.namedtuple(
        "InferModel",
        ("graph", "model", "placeholder_iterator", "placeholder_handle",
         "infer_iterator", "data_src_placeholder", "kb_placeholder",
         "batch_size_placeholder"))):
  pass


class SelfplayModel(
    collections.namedtuple(
        "SelfplayModel",
        ("graph", "model", "placeholder_iterator", "placeholder_handle",
         "train_iterator", "self_play_ft_iterator", "self_play_st_iterator",
         "data_placeholder", "kb_placeholder", "skip_count_placeholder",
         "batch_size_placeholder"))):
  pass


def create_train_model(model_creator,
                       hparams,
                       scope=None,
                       num_workers=1,
                       jobid=0,
                       extra_args=None):
  """Create graph, model and iterator for training."""
  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "train"):
    vocab_table, reverse_vocab_table = vocab_utils.create_vocab_tables(hparams.vocab_file)
    data_dataset = tf.data.TextLineDataset(hparams.train_data)
    kb_dataset = tf.data.TextLineDataset(hparams.train_kb)
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)
    # this is the actual train_iterator
    train_iterator = iterator_utils.get_iterator(
        data_dataset,
        kb_dataset,
        vocab_table,
        batch_size=hparams.batch_size,
        t1=hparams.t1.encode(),
        t2=hparams.t2.encode(),
        eod=hparams.eod,
        len_action=hparams.len_action,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        max_dialogue_len=hparams.max_dialogue_len,
        skip_count=skip_count_placeholder,
        num_shards=num_workers,
        shard_index=jobid)

    # this is the placeholder iterator. One can use this placeholder iterator
    # to switch between training and evauation.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_iterator.output_types, train_iterator.output_shapes)
    batched_iterator = iterator_utils.get_batched_iterator(iterator)
    model_device_fn = None
    if extra_args:
      model_device_fn = extra_args.model_device_fn
    with tf.device(model_device_fn):
      model = model_creator(
          hparams,
          iterator=batched_iterator,
          handle=handle,
          mode=tf.estimator.ModeKeys.TRAIN,
          vocab_table=vocab_table,
          scope=scope,
          extra_args=extra_args,
          reverse_vocab_table=reverse_vocab_table)
  return TrainModel(
      graph=graph,
      model=model,
      placeholder_iterator=iterator,
      train_iterator=train_iterator,
      placeholder_handle=handle,
      skip_count_placeholder=skip_count_placeholder)


def create_eval_model(model_creator, hparams, scope=None, extra_args=None):
  """Create train graph, model, src/tgt file holders, and iterator."""
  vocab_file = hparams.vocab_file
  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "eval"):
    vocab_table = vocab_utils.create_vocab_tables(vocab_file)[0]
    data_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    kb_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    data_dataset = tf.data.TextLineDataset(data_file_placeholder)
    kb_dataset = tf.data.TextLineDataset(kb_file_placeholder)
    # this is the eval_actual iterator
    eval_iterator = iterator_utils.get_iterator(
        data_dataset,
        kb_dataset,
        vocab_table,
        batch_size=hparams.batch_size,
        t1=hparams.t1.encode(),
        t2=hparams.t2.encode(),
        eod=hparams.eod,
        len_action=hparams.len_action,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        max_dialogue_len=hparams.max_dialogue_len)
    # this is the placeholder iterator
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, eval_iterator.output_types, eval_iterator.output_shapes)
    batched_iterator = iterator_utils.get_batched_iterator(iterator)

    model = model_creator(
        hparams,
        iterator=batched_iterator,
        handle=handle,
        mode=tf.estimator.ModeKeys.EVAL,
        vocab_table=vocab_table,
        scope=scope,
        extra_args=extra_args)

  return EvalModel(
      graph=graph,
      model=model,
      placeholder_iterator=iterator,
      placeholder_handle=handle,
      eval_iterator=eval_iterator,
      data_file_placeholder=data_file_placeholder,
      kb_file_placeholder=kb_file_placeholder)


def create_infer_model(model_creator, hparams, scope=None, extra_args=None):
  """Create inference model."""
  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "infer"):
    vocab_table, reverse_vocab_table = vocab_utils.create_vocab_tables(hparams.vocab_file)

    data_src_placeholder = tf.placeholder(
        shape=[None], dtype=tf.string, name="src_ph")
    kb_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name="kb_ph")
    batch_size_placeholder = tf.placeholder(
        shape=[], dtype=tf.int64, name="bs_ph")

    data_src_dataset = tf.data.Dataset.from_tensor_slices(data_src_placeholder)
    kb_dataset = tf.data.Dataset.from_tensor_slices(kb_placeholder)

    # this is the actual infer iterator
    infer_iterator = iterator_utils.get_infer_iterator(
        data_src_dataset,
        kb_dataset,
        vocab_table,
        batch_size=batch_size_placeholder,
        eod=hparams.eod,
        len_action=hparams.len_action)

    # this is the placeholder infer iterator
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, infer_iterator.output_types, infer_iterator.output_shapes)
    batched_iterator = iterator_utils.get_batched_iterator(iterator)

    model = model_creator(
        hparams,
        iterator=batched_iterator,
        handle=handle,
        mode=tf.estimator.ModeKeys.PREDICT,
        vocab_table=vocab_table,
        reverse_vocab_table=reverse_vocab_table,
        scope=scope,
        extra_args=extra_args)

  return InferModel(
      graph=graph,
      model=model,
      placeholder_iterator=iterator,
      placeholder_handle=handle,
      infer_iterator=infer_iterator,
      data_src_placeholder=data_src_placeholder,
      kb_placeholder=kb_placeholder,
      batch_size_placeholder=batch_size_placeholder)


#
def self_play_iterator_creator(hparams, num_workers, jobid):
  """create a self play iterator. There are iterators that will be created here.
  A supervised training iterator used for supervised learning. A full text
  iterator and structured iterator used for reinforcement learning self play.
  Full text iterators feeds data from text files while structured iterators
  are initialized directly from objects. The former one is used for traiing.
  The later one is used for self play dialogue generation to eliminate the
  need of serializing them into actual text
  files.
  """
  vocab_table = vocab_utils.create_vocab_tables(hparams.vocab_file)[0]
  data_dataset = tf.data.TextLineDataset(hparams.train_data)
  kb_dataset = tf.data.TextLineDataset(hparams.train_kb)
  skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)
  # this is the actual iterator for supervised training
  train_iterator = iterator_utils.get_iterator(
      data_dataset,
      kb_dataset,
      vocab_table,
      batch_size=hparams.batch_size,
      t1=hparams.t1.encode(),
      t2=hparams.t2.encode(),
      eod=hparams.eod,
      len_action=hparams.len_action,
      random_seed=hparams.random_seed,
      num_buckets=hparams.num_buckets,
      max_dialogue_len=hparams.max_dialogue_len,
      skip_count=skip_count_placeholder,
      num_shards=num_workers,
      shard_index=jobid)

  # this is the actual iterator for self_play_fulltext_iterator
  data_placeholder = tf.placeholder(
      shape=[None], dtype=tf.string, name="src_ph")
  kb_placeholder = tf.placeholder(shape=[None], dtype=tf.string, name="kb_ph")
  batch_size_placeholder = tf.placeholder(
      shape=[], dtype=tf.int64, name="bs_ph")

  dataset_data = tf.data.Dataset.from_tensor_slices(data_placeholder)
  kb_dataset = tf.data.Dataset.from_tensor_slices(kb_placeholder)

  self_play_fulltext_iterator = iterator_utils.get_infer_iterator(
      dataset_data,
      kb_dataset,
      vocab_table,
      batch_size=batch_size_placeholder,
      eod=hparams.eod,
      len_action=hparams.len_action,
      self_play=True)

  # this is the actual iterator for self_play_structured_iterator
  self_play_structured_iterator = tf.data.Iterator.from_structure(
      tf.data.get_output_types(self_play_fulltext_iterator),
      tf.data.get_output_shapes(self_play_fulltext_iterator))
  iterators = [
      train_iterator, self_play_fulltext_iterator, self_play_structured_iterator
  ]

  # this is the list of placeholders
  placeholders = [
      data_placeholder, kb_placeholder, batch_size_placeholder,
      skip_count_placeholder
  ]
  return iterators, placeholders


def create_selfplay_model(model_creator,
                          is_mutable,
                          num_workers,
                          jobid,
                          hparams,
                          scope=None,
                          extra_args=None):
  """create slef play models."""
  graph = tf.Graph()
  with graph.as_default(), tf.container(scope or "selfplay"):
    vocab_table, reverse_vocab_table = vocab_utils.create_vocab_tables(hparams.vocab_file)

    if is_mutable:
      mutable_index = 0
    else:
      mutable_index = 1

    # get a list of iterators and placeholders
    iterators, placeholders = self_play_iterator_creator(
        hparams, num_workers, jobid)
    train_iterator, self_play_fulltext_iterator, self_play_structured_iterator = iterators
    data_placeholder, kb_placeholder, batch_size_placeholder, skip_count_placeholder = placeholders

    # get an iterator handler
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, tf.data.get_output_types(train_iterator), tf.data.get_output_shapes(train_iterator))
    batched_iterator = iterator_utils.get_batched_iterator(iterator)

    model_device_fn = None
    if extra_args:
      model_device_fn = extra_args.model_device_fn
    with tf.device(model_device_fn):
      model = model_creator(
          hparams,
          iterator=batched_iterator,
          handle=handle,
          mode=[
              dialogue_utils.mode_self_play_mutable,
              dialogue_utils.mode_self_play_immutable
          ][mutable_index],
          vocab_table=vocab_table,
          reverse_vocab_table=reverse_vocab_table,
          scope=scope,
          extra_args=extra_args)
  return SelfplayModel(
      graph=graph,
      model=model,
      placeholder_iterator=iterator,
      placeholder_handle=handle,
      train_iterator=train_iterator,
      self_play_ft_iterator=self_play_fulltext_iterator,
      self_play_st_iterator=self_play_structured_iterator,
      data_placeholder=data_placeholder,
      kb_placeholder=kb_placeholder,
      skip_count_placeholder=skip_count_placeholder,
      batch_size_placeholder=batch_size_placeholder)


def create_emb_for_encoder_and_decoder(vocab_size,
                                       embed_size,
                                       dtype=tf.float32,
                                       num_partitions=0,
                                       scope=None):
  """Create embedding matrix for both encoder and decoder."""

  if num_partitions <= 1:
    partitioner = None
  else:
    # Note: num_partitions > 1 is required for distributed training due to
    # embedding_lookup tries to colocate single partition-ed embedding variable
    # with lookup ops. This may cause embedding variables being placed on worker
    # jobs.
    partitioner = tf.fixed_size_partitioner(num_partitions)

  with tf.variable_scope(
      scope or "embeddings", dtype=dtype, partitioner=partitioner) as scope:
    # Share embedding
    embedding_encoder = tf.get_variable("shared_embedding",
                                        [vocab_size, embed_size], dtype)
    embedding_decoder = embedding_encoder

  return embedding_encoder, embedding_decoder


def _single_cell(num_units,
                 dropout,
                 mode,
                 residual_connection=False,
                 device_str=None):
  """Create an instance of a single RNN cell."""
  dropout = dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0

  # Cell Type
  utils.print_out("  GRU", new_line=False)
  single_cell = tf.nn.rnn_cell.GRUCell(num_units)

  # Dropout (= 1 - keep_prob)
  if dropout > 0.0:
    single_cell = tf.nn.rnn_cell.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))
    utils.print_out(
        "  %s, dropout=%g " % (type(single_cell).__name__, dropout),
        new_line=False)

  # Residual
  if residual_connection:
    single_cell = tf.nn.rnn_cell.ResidualWrapper(single_cell)
    utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

  # Device Wrapper
  if device_str:
    single_cell = tf.nn.rnn_cell.DeviceWrapper(single_cell, device_str)
    utils.print_out(
        "  %s, device=%s" % (type(single_cell).__name__, device_str),
        new_line=False)

  return single_cell


def _cell_list(num_units,
               num_layers,
               num_residual_layers,
               dropout,
               mode,
               num_gpus,
               base_gpu=0,
               single_cell_fn=None):
  """Create a list of RNN cells."""
  if not single_cell_fn:
    single_cell_fn = _single_cell

  # Multi-GPU
  cell_list = []
  for i in range(num_layers):
    utils.print_out("  cell %d" % i, new_line=False)
    single_cell = single_cell_fn(
        num_units=num_units,
        dropout=dropout,
        mode=mode,
        residual_connection=(i >= num_layers - num_residual_layers),
        device_str=get_device_str(i + base_gpu, num_gpus),
    )
    utils.print_out("")
    cell_list.append(single_cell)

  return cell_list


def create_rnn_cell(num_units,
                    num_layers,
                    num_residual_layers,
                    dropout,
                    mode,
                    num_gpus,
                    base_gpu=0,
                    single_cell_fn=None,
                    all_layer_outputs=False):
  """Create multi-layer RNN cell. When all_layer_outputs is True, that means we
  want hidden states of all timestamps to pass through. In this case we use
  MultiRNNCell, a slightly modified tensorflow RNN cell.
  """
  cell_list = _cell_list(
      num_units=num_units,
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      dropout=dropout,
      mode=mode,
      num_gpus=num_gpus,
      base_gpu=base_gpu,
      single_cell_fn=single_cell_fn)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    print(all_layer_outputs, "all_layer_outputs")

    if all_layer_outputs:
      return MultiRNNCell(cell_list)
    else:
      return tf.nn.rnn_cell.MultiRNNCell(cell_list)


def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

  return clipped_gradients, gradient_norm_summary


def get_variables_available_in_checkpoint(variables,
                                          ckpt,
                                          include_global_step=True):
  if isinstance(variables, list):
    variable_names_map = {variable.op.name: variable for variable in variables}
  elif isinstance(variables, dict):
    variable_names_map = variables
  else:
    raise ValueError("`variables` is expected to be a list or dict.")

  ckpt_reader = tf.train.NewCheckpointReader(ckpt)
  ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
  if include_global_step:
    ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)

  vars_in_ckpt = {}
  for variable_name, variable in sorted(variable_names_map.items()):
    variable_name_without_partition = re.sub("/part_[0-9]+$", "", variable_name)
    if variable_name in ckpt_vars_to_shape_map:
      if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
        vars_in_ckpt[variable_name] = variable
      else:
        logging.warning(
            "Variable [%s] is available in checkpoint, but has an "
            "incompatible shape with model variable. Checkpoint "
            "shape: [%s], model variable shape: [%s]. This "
            "variable will not be initialized from the checkpoint.",
            variable_name, ckpt_vars_to_shape_map[variable_name],
            variable.shape.as_list())
    elif variable_name_without_partition in ckpt_vars_to_shape_map:
      # Do not check shape for partition variables
      vars_in_ckpt[variable_name] = variable
    else:
      logging.warning("Variable [%s] is not available in checkpoint",
                      variable_name)

  # It seems the restore does something smart about partitioned variables.
  # Should keep it as a list instead of using partitioned variable keys.
  if isinstance(variables, list):
    return list(vars_in_ckpt.values())
  return vars_in_ckpt


def load_model(model, ckpt, session, name):
  start_time = time.time()
  available_var_list = (
      get_variables_available_in_checkpoint(model.saver._var_list, ckpt))
  # TODO: handle verbosity
  # logging.info("available_var_list:%s,%s", len(available_var_list),
  #             available_var_list)
  tf.train.Saver(available_var_list).restore(session, ckpt)
  session.run(tf.tables_initializer())
  utils.print_out("  loaded %s model parameters from %s, time %.2fs" %
                  (name, ckpt, time.time() - start_time))
  return model


def full_restore(session, ckpt):
  start_time = time.time()
  available_var_list = (
      get_variables_available_in_checkpoint(tf.global_variables(), ckpt))
  logging.info("available_var_list:%s,%s", len(available_var_list),
               available_var_list)
  tf.train.Saver(available_var_list).restore(session, ckpt)
  session.run(tf.tables_initializer())
  utils.print_out(
      "full restore from %s, time %.2fs" % (ckpt, time.time() - start_time))


def create_or_load_model(model, model_dir, session, name):
  """Create translation model and initialize or load parameters in session."""
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    start_time = time.time()
    # It only takes a few seconds to initialize all variables.
    session.run(tf.global_variables_initializer())
    logging.info(
        "Initialize %s model with fresh parameters before loading variables "
        "from the checkpoint, time %.2fs", name,
        time.time() - start_time)
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

  global_step = model.global_step.eval(session=session)
  return model, global_step


def compute_perplexity(model, sess, name, eval_handle):
  """Compute perplexity of the output of the model based on loss function."""

  def aggregate_all_summaries(original, updates):
    for key in updates:
      if key not in original:
        original[key] = 0.0
      original[key] += updates[key]
    return original

  total_loss = 0
  total_predict_count = 0
  start_time = time.time()
  aggregated_summaries = {}
  batch_processed = 0
  while True:
    try:
      loss, all_summaries, predict_count, batch_size = model.eval(
          sess, eval_handle)
      total_loss += loss * batch_size
      batch_processed += 1
      total_predict_count += predict_count
      aggregated_summaries = aggregate_all_summaries(aggregated_summaries,
                                                     all_summaries)
    except tf.errors.OutOfRangeError:
      break

  perplexity = utils.safe_exp(total_loss / total_predict_count)
  for key in aggregated_summaries:
    if key not in set(
        ["eval_dialogue_loss1", "eval_dialogue_loss2", "eval_action_loss3"]):
      aggregated_summaries[key] /= batch_processed
  utils.print_time("  eval %s: perplexity %.2f" % (name, perplexity),
                   start_time)
  return perplexity, aggregated_summaries
