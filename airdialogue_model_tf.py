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
"""This file is the main entry for the airdialogue model."""

import argparse
from functools import partial
import os
import random
import sys
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.contrib as contrib
import inference
import self_play
import train
from utils import misc_utils as utils
from utils import vocab_utils
from utils.dialogue_utils import task_INFER
from utils.dialogue_utils import task_SP_DISTRIBUTED
from utils.dialogue_utils import task_SP_EVAL
from utils.dialogue_utils import task_TRAINEVAL

FLAGS = None


def add_arguments(parser):
  """Add argumentsf from the parser."""
  parser.register("type", "bool", lambda v: v.lower() == "true")

  # evaluation
  parser.add_argument(
      "--eval_forever",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="""If enabled, we will do evaluation forever once new checkpoints
              are arrived. This includes both self-play evaluation and
              inference.""")

  # architecture
  parser.add_argument(
      "--master",
      type=str,
      default="",
      help="Name of the Tensorflow master to use.")
  parser.add_argument(
      "--rl_training",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help="Whether we do rl training in SP_distributed.")
  parser.add_argument(
      "--train_reward_type",
      type=str,
      default="scaled",
      help="which reward type to use when training. scaled|discrete|combined|extreme"
  )
  parser.add_argument(
      "--identity", type=str, default="", help="The identity of the instance")
  parser.add_argument(
      "--num_units",
      type=int,
      default=32,
      help="unit size of the seq2seq model.")
  parser.add_argument(
      "--num_layers",
      type=int,
      default=2,
      help="number of layers of the seq2seq model.")
  parser.add_argument(
      "--residual",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Whether to add residual connections.")
  parser.add_argument(
      "--num_embeddings_partitions",
      type=int,
      default=0,
      help="Number of partitions for embedding vars.")

  # optimizer
  parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
  parser.add_argument(
      "--learning_rate",
      type=float,
      default=1.0,
      help="Learning rate. Adam: 0.001 | 0.0001")

  parser.add_argument(
      "--learning_rate_warmup_steps",
      type=int,
      default=0,
      help="How many steps we inverse-decay learning.")
  parser.add_argument(
      "--learning_rate_warmup_factor",
      type=float,
      default=1.0,
      help="The inverse decay factor for each warmup step.")
  parser.add_argument(
      "--start_decay_step", type=int, default=0, help="When we start to decay")
  parser.add_argument(
      "--decay_steps", type=int, default=10000, help="How frequent we decay")
  parser.add_argument(
      "--decay_factor", type=float, default=0.98, help="How much we decay.")
  parser.add_argument(
      "--num_train_steps", type=int, default=12000, help="Num steps to train.")
  parser.add_argument(
      "--colocate_gradients_with_ops",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help=("Whether try colocating gradients with "
            "corresponding op"))

  # initializer
  parser.add_argument(
      "--init_op",
      type=str,
      default="uniform",
      help="uniform | glorot_normal | glorot_uniform")
  parser.add_argument(
      "--init_weight",
      type=float,
      default=0.1,
      help=("for uniform init_op, initialize weights "
            "between [-this, this]."))

  parser.add_argument(
      "--out_dir",
      type=str,
      default="./data/out_dir",
      help="Store log/model files.")

  parser.add_argument(
      "--dropout", type=float, default=0.2, help="Dropout rate (not keep_prob)")
  parser.add_argument(
      "--max_gradient_norm",
      type=float,
      default=5.0,
      help="Clip gradients to this norm.")
  parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")

  parser.add_argument(
      "--steps_per_stats",
      type=int,
      default=2,
      help=("How many training steps to do per stats logging."
            "Save checkpoint every 10x steps_per_stats"))
  parser.add_argument(
      "--num_buckets",
      type=int,
      default=5,
      help="Put data into similar-length buckets.")

  # Misc
  parser.add_argument(
      "--num_gpus", type=int, default=1, help="Number of gpus in each worker.")
  parser.add_argument(
      "--log_device_placement",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Debug GPU allocation.")
  parser.add_argument(
      "--metrics",
      type=str,
      default="bleu",
      help=("Comma-separated list of evaluations "
            "metrics (bleu,rouge,accuracy)"))

  parser.add_argument(
      "--scope", type=str, default=None, help="scope to put variables under")
  parser.add_argument(
      "--hparams_path",
      type=str,
      default=None,
      help=("Path to standard hparams json file that overrides"
            "hparams values from FLAGS."))
  parser.add_argument(
      "--random_seed",
      type=int,
      default=None,
      help="Random seed (>0, set a specific seed).")
  parser.add_argument(
      "--override_loaded_hparams",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Override loaded hparams with values specified")

  # Inference
  parser.add_argument(
      "--inference_input_file",
      type=str,
      default=None,
      help="Set to the text to decode.")

  parser.add_argument(
      "--infer_batch_size",
      type=int,
      default=128,
      help="Batch size for inference mode.")

  parser.add_argument(
      "--beam_width",
      type=int,
      default=0,
      help=("""\
      beam width when using beam search decoder. If 0 (default), use standard
      decoder with greedy helper.\
      """))
  parser.add_argument(
      "--length_penalty_weight",
      type=float,
      default=0.0,
      help="Length penalty for beam search.")

  # Job info
  parser.add_argument(
      "--jobid", type=int, default=0, help="Task id of the worker.")
  parser.add_argument(
      "--num_workers",
      type=int,
      default=1,
      help="Number of workers (inference only).")

  # dialogue
  parser.add_argument(
      "--input_dir",
      type=str,
      default=None,
      help="""if set, following arguments will be ignored and the pathes of the
              input files will be automatically determined.
              train_data, train_kb
              dev_data, dev_kb
              test_data, test_kb,
              infer_src_data, infer_tar_data, infer_kb,
              self_play_train_data, self_play_train_kb
              self_play_eval_data, self_play_eval_kb
              vocab_file
              """)
  # train
  parser.add_argument(
      "--train_data",
      type=str,
      default=None,
      help="dialogue data from train set")
  parser.add_argument(
      "--train_kb", type=str, default=None, help="kb from train set")
  # selfplay train
  parser.add_argument(
      "--self_play_train_data",
      type=str,
      default=None,
      help="dialogue data for self-play training")
  parser.add_argument(
      "--self_play_train_kb",
      type=str,
      default=None,
      help="kb for self-play training")
  # dev
  parser.add_argument(
      "--dev_data",
      type=str,
      default=None,
      help="dialogue data from dev set for evaluation")
  parser.add_argument(
      "--dev_kb", type=str, default=None, help="kb from dev set for evaluation")
  parser.add_argument(
      "--infer_src_data",
      type=str,
      default=None,
      help="dialogue data source for inference")
  parser.add_argument(
      "--infer_tar_data",
      type=str,
      default=None,
      help="dialogue data target for inference")
  parser.add_argument(
      "--codalab",
      action="store_true",
      help="""Indicates if working with Codalab workflow. Generally decreases
        unecessary steps (like self inference scoring)""")
  parser.add_argument(
      "--infer_kb", type=str, default=None, help="kb for inference")
  parser.add_argument(
      "--self_play_eval_data",
      type=str,
      default=None,
      help="dialogue data for self-play evaluation")
  parser.add_argument(
      "--self_play_eval_kb",
      type=str,
      default=None,
      help="kb for self-play evaluation")
  parser.add_argument(
      "--vocab_file", type=str, default=None, help="vocabulary file")
  parser.add_argument(
      "--inference_output_file",
      type=str,
      default=None,
      help="""Output file for dialogue inference. If not set, it will be set to
              inference_out.txt under out_dir.""")
  parser.add_argument(
      "--selfplay_eval_output_file",
      type=str,
      default=None,
      help="""Output file for dialogue selfplay evaluation. If not set, it will be set to
              selfplay_eval_out.txt under out_dir.""")
  parser.add_argument(
      "--eval_prefix",
      type=str,
      default=None,
      help="prefix of the evaluation dataset.")

  parser.add_argument(
      "--max_dialogue_len",
      type=int,
      default=400,
      help="maximum length for a dialogue during training")
  parser.add_argument(
      "--max_inference_len",
      type=int,
      default=50,
      help="maximum sentence length for dialogue inference")
  parser.add_argument(
      "--self_play_start_turn",
      type=str,
      default=None,
      help="Force self-play to run for an agent/customer start. [agent | customer]"
  )
  parser.add_argument(
      "--num_kb_fields_per_entry",
      type=int,
      default=13,
      help="number of attributes of each flight in the knowledge base")
  parser.add_argument(
      "--len_action",
      type=int,
      default=4,
      help="number of dialogue states for each conversation")
  # selfplay
  parser.add_argument(
      "--self_play_pretrain_dir",
      type=str,
      default=None,
      help="the directory for a pre-trained model used to initialize self-play. This is usually the supervised learning model."
  )
  parser.add_argument(
      "--max_dialogue_turns",
      type=int,
      default=50,
      help="The maximum number of turns that a dialogue would take to terminal. When conducting self-play, this is the maximum number of turns we expect the dialogue to reach an end-of-dialogue token."
  )
  parser.add_argument(
      "--train_threadhold",
      type=int,
      default=2,
      help="we won't train the model until train_threadhold samples are reached"
  )
  # reinforcement learning
  parser.add_argument(
      "--reward_discount",
      type=float,
      default=0.95,
      help="reward discount in RL")
  parser.add_argument(
      "--task_type",
      type=str,
      default=task_TRAINEVAL,
      help="the type of the task that is being conducted. It has to be one of TRAINEVAL|INFER|SP_EVAL|SP_DISTRIBUTED"
  )
  parser.add_argument(
      "--self_play_batch_size",
      type=int,
      default=10,
      help="batch size for self-play dialogue generation")  # talk batch size
  parser.add_argument(
      "--self_play_update_batch_size",
      type=int,
      default=10,
      help="batch size for self-play dialogue update")
  parser.add_argument(
      "--self_play_eval_batch_size",
      type=int,
      default=10,
      help="batch size for self-play evaluation")
  # control flow
  parser.add_argument(
      "--debug",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="a flag reserved for debug purpose")
  # new parameters
  parser.add_argument(
      "--unit_value_network",
      type=int,
      default=32,
      help="number of units for the value network")
  parser.add_argument(
      "--layer_value_network",
      type=int,
      default=2,
      help="number of layers for the value network")
  parser.add_argument(
      "--encoder_kb_unit",
      type=int,
      default=32,
      help="number of units for the knowledge base encoder")
  parser.add_argument(
      "--encoder_intent_unit",
      type=int,
      default=32,
      help="number of units for the intent encoder")

  # distribution self play training
  parser.add_argument(
      "--num_self_play_train_steps",
      type=int,
      default=12000,
      help="Num selfplay steps to train.")

  parser.add_argument(
      "--ps_tasks",
      type=int,
      default=0,
      help="Number of parameter servers for distributed training.")
  parser.add_argument(
      "--startup_delay_steps",
      type=int,
      default=500,
      help="Delay worker replica startup incrementally by this many steps.")
  parser.add_argument(
      "--immutable_model_reload_freq",
      type=int,
      default=100,
      help="how often we reload the immutable model during self-play training")
  parser.add_argument(
      "--self_play_dist_save_freq",
      type=int,
      default=5,
      help="how often we save the model during self-play training")
  parser.add_argument("--self_play_loss_method", type=int, default=1, help="")
  parser.add_argument(
      "--self_play_variable_method",
      type=int,
      default=2,
      help="0-only aux, 1- only seq2seq, 2-both aux and seq")
  parser.add_argument(
      "--self_play_sl_multiplier",
      type=int,
      default=1,
      help="how many supervised updates we do for each self-play update")
  parser.add_argument(
      "--self_play_immutable_gpu",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="whether we use GPU for immutable agent when conducting self-play")
  parser.add_argument(
      "--learning_rate2",
      type=float,
      default=0.001,
      help="Learning rate for the first speaker when doing self-play updates. Adam: 0.001 | 0.0001"
  )
  parser.add_argument(
      "--learning_rate3",
      type=float,
      default=0.001,
      help="Learning rate for the second speaker when doing self-play updates. Adam: 0.001 | 0.0001"
  )
  parser.add_argument(
      "--max_gradient_norm2",
      type=float,
      default=1,
      help="gradient norm for the policy network of the first speaker when doing self-play updates"
  )
  parser.add_argument(
      "--max_gradient_norm3",
      type=float,
      default=1,
      help="gradient norm for the policy network of the second speaker when doing self-play updates"
  )


def create_hparams(flags):
  """Create training hparams."""
  return contrib.training.HParams(
      # Data
      input_dir=flags.input_dir,
      out_dir=flags.out_dir,
      eval_prefix=flags.eval_prefix,
      # architecture
      num_units=flags.num_units,
      num_layers=flags.num_layers,
      dropout=flags.dropout,
      residual=flags.residual,
      num_embeddings_partitions=flags.num_embeddings_partitions,

      # Train
      optimizer=flags.optimizer,
      num_train_steps=flags.num_train_steps,
      batch_size=flags.batch_size,
      init_op=flags.init_op,
      init_weight=flags.init_weight,
      max_gradient_norm=flags.max_gradient_norm,
      learning_rate=flags.learning_rate,
      learning_rate2=flags.learning_rate2,
      learning_rate3=flags.learning_rate3,
      learning_rate_warmup_steps=flags.learning_rate_warmup_steps,
      learning_rate_warmup_factor=flags.learning_rate_warmup_factor,
      start_decay_step=flags.start_decay_step,
      decay_factor=flags.decay_factor,
      decay_steps=flags.decay_steps,
      colocate_gradients_with_ops=flags.colocate_gradients_with_ops,

      # Data constraints
      num_buckets=flags.num_buckets,

      # Inference
      infer_batch_size=flags.infer_batch_size,
      beam_width=flags.beam_width,
      length_penalty_weight=flags.length_penalty_weight,

      # Misc
      num_gpus=flags.num_gpus,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=flags.steps_per_stats,
      metrics=flags.metrics.split(","),
      log_device_placement=flags.log_device_placement,
      random_seed=flags.random_seed,
      override_loaded_hparams=flags.override_loaded_hparams,
      # dialogue
      start_of_turn1=vocab_utils.start_of_turn1,
      start_of_turn2=vocab_utils.start_of_turn2,
      end_of_dialogue=vocab_utils.end_of_dialogue,
      # train
      train_data=flags.train_data,
      train_kb=flags.train_kb,
      self_play_train_data=flags.self_play_train_data,
      self_play_train_kb=flags.self_play_train_kb,
      # dev
      dev_data=flags.dev_data,
      dev_kb=flags.dev_kb,

      # inference data
      infer_src_data=flags.infer_src_data,
      infer_tar_data=flags.infer_tar_data,
      infer_kb=flags.infer_kb,
      # selfplay evaluation data
      self_play_eval_data=flags.self_play_eval_data,
      self_play_eval_kb=flags.self_play_eval_kb,
      vocab_file=flags.vocab_file,
      max_dialogue_len=flags.max_dialogue_len,
      max_inference_len=flags.max_inference_len,
      self_play_start_turn=flags.self_play_start_turn,
      num_kb_fields_per_entry=flags.num_kb_fields_per_entry,
      len_action=flags.len_action,
      # selfplay
      self_play_pretrain_dir=flags.self_play_pretrain_dir,
      max_dialogue_turns=flags.max_dialogue_turns,
      train_threadhold=flags.train_threadhold,
      # reinforcement learning
      reward_discount=flags.reward_discount,
      unit_value_network=flags.unit_value_network,
      layer_value_network=flags.layer_value_network,
      encoder_kb_unit=flags.encoder_kb_unit,
      encoder_intent_unit=flags.encoder_intent_unit,
      self_play_batch_size=flags.self_play_batch_size,
      self_play_update_batch_size=flags.self_play_update_batch_size,
      self_play_eval_batch_size=flags.self_play_eval_batch_size,
      # others
      debug=flags.debug,
      inference_output_file=flags.inference_output_file,
      selfplay_eval_output_file=flags.selfplay_eval_output_file,
      task_type=flags.task_type,
      codalab=flags.codalab,
      # self-play
      num_self_play_train_steps=flags.num_self_play_train_steps,
      immutable_model_reload_freq=flags.immutable_model_reload_freq,
      self_play_dist_save_freq=flags.self_play_dist_save_freq,
      self_play_loss_method=flags.self_play_loss_method,
      self_play_variable_method=flags.self_play_variable_method,
      self_play_sl_multiplier=flags.self_play_sl_multiplier,
      self_play_immutable_gpu=flags.self_play_immutable_gpu,
      max_gradient_norm3=flags.max_gradient_norm3,
      max_gradient_norm2=flags.max_gradient_norm2,
      train_reward_type=flags.train_reward_type,
      rl_training=flags.rl_training,
      # others
      identity=flags.identity,
      eval_forever=flags.eval_forever)


def process_input_path(hparams):
  # if input_dir is set, we ignore individual pathes to input files
  if hparams.input_dir:
    # train
    if not hparams.train_data:
      hparams.train_data = os.path.join(hparams.input_dir, "train.data")
    if not hparams.train_kb:
      hparams.train_kb = os.path.join(hparams.input_dir, "train.kb")
    if not hparams.self_play_train_data:
      hparams.self_play_train_data = os.path.join(hparams.input_dir,
                                                  "train.selfplay.data")
    if not hparams.self_play_train_kb:
      hparams.self_play_train_kb = os.path.join(hparams.input_dir,
                                                "train.selfplay.kb")
    # dev
    if not hparams.dev_data:
      hparams.dev_data = os.path.join(hparams.input_dir, "dev.eval.data")
    if not hparams.dev_kb:
      hparams.dev_kb = os.path.join(hparams.input_dir, "dev.eval.kb")
    if not hparams.vocab_file:
      hparams.vocab_file = os.path.join(hparams.input_dir, "vocab.txt")

    if hparams.task_type == task_INFER:
      if not hparams.infer_src_data:
        hparams.infer_src_data = os.path.join(
            hparams.input_dir, hparams.eval_prefix + ".infer.src.data")
      if not hparams.infer_tar_data:
        hparams.infer_tar_data = os.path.join(
            hparams.input_dir, hparams.eval_prefix + ".infer.tar.data")
      if not hparams.infer_kb:
        hparams.infer_kb = os.path.join(hparams.input_dir,
                                        hparams.eval_prefix + ".infer.kb")
    if hparams.task_type == task_SP_EVAL:
      if not (hparams.self_play_eval_data and hparams.self_play_eval_kb):
        hparams.self_play_eval_data = os.path.join(
            hparams.input_dir, hparams.eval_prefix + ".selfplay.eval.data")
        hparams.self_play_eval_kb = os.path.join(
            hparams.input_dir, hparams.eval_prefix + ".selfplay.eval.kb")
  if hparams.codalab:
    hparams.infer_tar_data = None
  if hparams.task_type == task_INFER and (not hparams.inference_output_file):
    hparams.inference_output_file = os.path.join(hparams.out_dir,
                                                 "inference_out.txt")
  if hparams.task_type == task_SP_EVAL and (
      not hparams.selfplay_eval_output_file):
    hparams.selfplay_eval_output_file = os.path.join(hparams.out_dir,
                                                     "selfplay_eval_out.txt")

  # set flags for tensorboard on infer and selfplay eval tasks
  if (not hparams.identity) and hparams.task_type in [task_SP_EVAL, task_INFER]:
    mapping = {task_INFER: "infer", task_SP_EVAL: "selfplay"}
    hparams.identity = mapping[hparams.task_type] + "_" + hparams.eval_prefix
    print("hparams.identity", hparams.identity)
  return hparams


def extend_hparams(hparams):
  """Extend training hparams."""

  # Set num_residual_layers
  if hparams.residual and hparams.num_layers > 1:
    num_residual_layers = hparams.num_layers - 1
  else:
    num_residual_layers = 0
  hparams.add_hparam("num_residual_layers", num_residual_layers)
  print("hparams.vocab_file", hparams.vocab_file)
  hparams.add_hparam("vocab_size",
                     vocab_utils.get_vocab_size(hparams.vocab_file))
  hparams.add_hparam("t1", vocab_utils.start_of_turn1)
  hparams.add_hparam("t2", vocab_utils.start_of_turn2)
  hparams.add_hparam("eod", vocab_utils.end_of_dialogue)
  hparams.add_hparam("unk", vocab_utils.UNK)

  # Check out_dir
  if not tf.io.gfile.exists(hparams.out_dir):
    utils.print_out("# Creating output directory %s ..." % hparams.out_dir)
    tf.gfile.MakeDirs(hparams.out_dir)
  # Evaluation
  for metric in hparams.metrics:
    hparams.add_hparam("best_" + metric, 0)  # larger is better
    best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
    hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
    tf.gfile.MakeDirs(best_metric_dir)

  # path
  if not hparams.inference_output_file:
    # If not set, it will be set to inference_out.txt under variable out_dir
    hparams.inference_output_file = os.path.join(hparams.out_dir,
                                                 "inference_out.txt")
  return hparams


def ensure_compatible_hparams(hparams, default_hparams, hparams_path):
  """Make sure the loaded hparams is compatible with new changes.

  For
  compatible reason, if there are new fields in default_hparams, we add
  them to the current hparams.
  """

  default_hparams = utils.maybe_parse_standard_hparams(default_hparams,
                                                       hparams_path)
  default_config = default_hparams.values()
  config = hparams.values()
  for key in default_config:
    if key not in config:
      hparams.add_hparam(key, default_config[key])

  # Make sure that the loaded model has latest values for the below keys
  updated_keys = [
      "out_dir", "num_gpus", "beam_width", "length_penalty_weight",
      "num_train_steps", "train_data", "train_kb", "dev_data", "dev_kb",
      "infer_src_data", "infer_tar_data", "infer_kb", "self_play_eval_data",
      "self_play_eval_kb", "self_play_train_data", "self_play_train_kb",
      "vocab_file", "max_dialogue_len", "max_inference_len",
      "num_kb_fields_per_entry", "len_action", "self_play_model_dir",
      "max_dialogue_turns", "train_threadhold", "reward_discount",
      "do_selfplay", "self_play_batch_size", "self_play_update_batch_size",
      "self_play_eval_batch_size", "inference_output_file", "task_type",
      "self_play_pretrain_dir", "learning_rate", "colocate_gradients_with_ops",
      "immutable_model_reload_freq", "optimizer", "self_play_loss_method",
      "self_play_variable_method", "self_play_sl_multiplier", "batch_size",
      "log_device_placement", "metrics", "self_play_immutable_gpu",
      "learning_rate2", "learning_rate3", "infer_batch_size", "steps_per_stats",
      "train_reward_type", "rl_training", "dev_infer_src_data",
      "dev_infer_tar_data", "dev_infer_kb", "dev_self_play_eval_data",
      "dev_self_play_eval_kb", "test_infer_src_data", "test_infer_tar_data",
      "test_infer_kb", "test_self_play_eval_data", "test_self_play_eval_kb",
      "eval_prefix", "eval_forever", "selfplay_eval_output_file",
      "num_self_play_train_steps", "codalab"
  ]
  for key in updated_keys:
    if key in default_config and getattr(hparams, key) != default_config[key]:
      utils.print_out(
          "# Updating hparams.%s: %s -> %s" %
          (key, str(getattr(hparams, key)), str(default_config[key])))
      setattr(hparams, key, default_config[key])
  return hparams


def create_or_load_hparams(load_dir, default_hparams, hparams_path,
                           save_hparams):
  """Create hparams or load hparams from out_dir."""
  hparams = utils.load_hparams(load_dir)
  if not hparams:
    hparams = default_hparams
    # Override hparams values with existing standard hparams config
    hparams = utils.maybe_parse_standard_hparams(hparams, hparams_path)
    hparams = process_input_path(hparams)
    hparams = extend_hparams(hparams)
  else:
    hparams = ensure_compatible_hparams(hparams, default_hparams, hparams_path)
    hparams = process_input_path(hparams)

  # Save HParams
  if save_hparams:
    utils.save_hparams(default_hparams.out_dir, hparams)
    for metric in hparams.metrics:
      utils.save_hparams(getattr(hparams, "best_" + metric + "_dir"), hparams)

  # Print HParams
  utils.print_hparams(hparams)
  return hparams


def set_random_seed(flags, jobid):
  random_seed = flags.random_seed
  if random_seed is not None and random_seed > 0:
    utils.print_out("# Set random seed to %d" % random_seed)
    random.seed(random_seed + jobid)
    np.random.seed(random_seed + jobid)


def setup_exps(out_dir):
  """Create output folder."""
  if not tf.gfile.Exists(out_dir):
    tf.gfile.MakeDirs(out_dir)
  return out_dir


def load_hparams(flags, default_hparams, save_hparams):
  if flags.task_type not in [task_SP_EVAL, task_SP_DISTRIBUTED]:
    # supervised models load from out_dir.
    load_dir = flags.out_dir
  else:
    # self-play models load from self_play_pretrain_dir.
    load_dir = flags.self_play_pretrain_dir
  hparams = create_or_load_hparams(
      load_dir, default_hparams, flags.hparams_path, save_hparams=save_hparams)
  return hparams


def main_simple(unused_argv):
  flags = FLAGS
  default_hparams = create_hparams(flags)  #  create default hparams
  utils.print_out("# Job id %d" % flags.jobid)
  set_random_seed(flags, flags.jobid)
  flags.out_dir = setup_exps(flags.out_dir)

  print("flags.task_type", flags.task_type)
  if flags.task_type == task_TRAINEVAL:
    work_fn = train.train
    save_hparams = True
  elif flags.task_type == task_INFER:
    work_fn = inference.infer_fn
    save_hparams = False
  elif flags.task_type == task_SP_EVAL:
    work_fn = self_play.self_play_eval_fn
    save_hparams = False
  elif flags.task_type == task_SP_DISTRIBUTED:
    work_fn = partial(
        self_play.multi_worker_selfplay,
        is_chief=(flags.jobid == 0),
        ps_tasks=flags.ps_tasks,
        num_workers=flags.num_workers,
        jobid=flags.jobid,
        startup_delay_steps=flags.startup_delay_steps)
    save_hparams = (flags.jobid == 0)
  else:
    print("task type=", flags.task_type)
    raise ValueError("invalid task type")

  # construct and possibly save hparams
  hparams = load_hparams(flags, default_hparams, save_hparams=save_hparams)

  machine_identity = hparams.identity
  utils.print_out("the identity of the machine is " + machine_identity)
  # run work_fn
  work_fn(hparams, machine_identity, target_session=FLAGS.master)


if __name__ == "__main__":
  diag_parser = argparse.ArgumentParser()
  add_arguments(diag_parser)
  FLAGS, unparsed = diag_parser.parse_known_args()
  tf.app.run(main=main_simple, argv=[sys.argv[0]] + unparsed)
