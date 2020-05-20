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

"""Predict the next sentence given a partial human dialogue."""

import os
import time
import tensorflow as tf
import model as diag_model
import model_helper
from utils import dialogue_utils
from utils import misc_utils as utils
from utils.dialogue_utils import _sample_decode
from utils.dialogue_utils import load_data


def _internal_eval(model, global_step, sess, real_iterator, iterator_feed_dict,
                   iterator_handle, summary_writer, label):
  """Computing perplexity."""
  sess.run(real_iterator.initializer, feed_dict=iterator_feed_dict)
  ppl, all_summaries = model_helper.compute_perplexity(model, sess, label,
                                                       iterator_handle)

  utils.add_summary(summary_writer, global_step, "%s_ppl" % label, ppl)
  for key in all_summaries:
    utils.add_summary(summary_writer, global_step, key, all_summaries[key])
  return ppl


def run_internal_eval(eval_model, eval_handle, eval_sess, model_dir, hparams,
                      summary_writer):
  """Compute internal evaluation (perplexity) for dev."""
  with eval_model.graph.as_default():
    loaded_eval_model, global_step = model_helper.create_or_load_model(
        eval_model.model, model_dir, eval_sess, "eval")

  dev_eval_iterator_feed_dict = {
      eval_model.data_file_placeholder: hparams.dev_data,
      eval_model.kb_file_placeholder: hparams.dev_kb,
  }

  dev_ppl = _internal_eval(
      loaded_eval_model, global_step, eval_sess, eval_model.eval_iterator,
      dev_eval_iterator_feed_dict, eval_handle, summary_writer, "dev")
  return dev_ppl, None


def single_worker_inference(infer_model, infer_sess, eval_model, eval_sess,
                            ckpt, summary_writer, global_step, hparams):
  """the actual function for inference."""
  # load datasets
  infer_src_data = load_data(hparams.infer_src_data)
  if hparams.infer_tar_data:
    infer_tar_data = load_data(hparams.infer_tar_data)
  else:
    infer_tar_data = None
  infer_kb = load_data(hparams.infer_kb)

  # load model and session
  start_time = time.time()
  with infer_model.graph.as_default():
    loaded_infer_model = model_helper.load_model(infer_model.model, ckpt,
                                                 infer_sess, "infer")
    infer_sess.run(
        infer_model.infer_iterator.initializer,
        feed_dict={
            infer_model.data_src_placeholder: infer_src_data,
            infer_model.kb_placeholder: infer_kb,
            infer_model.batch_size_placeholder: hparams.infer_batch_size
        })
    infer_handle = infer_sess.run(infer_model.infer_iterator.string_handle())

    # Decode
    utils.print_out("# Start decoding")
    evaluation_scores = dialogue_utils.decode_and_evaluate(
        "infer",
        loaded_infer_model,
        infer_handle,
        infer_sess,
        hparams.inference_output_file,
        ref_file=hparams.infer_tar_data,
        metrics=hparams.metrics,
        hparams=hparams,
        infer_src_data=infer_src_data)
    # summary writer
    for key in evaluation_scores:
      # utils.add_summary(summary_writer,)
      utils.add_summary(summary_writer, global_step, key,
                        evaluation_scores[key])
    # sample some dialogue and decode them for qualitative examination
    _sample_decode(loaded_infer_model, global_step, infer_handle, infer_sess,
                   hparams, infer_model.infer_iterator, infer_src_data,
                   infer_tar_data, infer_kb, infer_model.data_src_placeholder,
                   infer_model.kb_placeholder,
                   infer_model.batch_size_placeholder)
  # run eval model to get perplexity
  if not hparams.codalab and hparams.infer_tar_data:
    eval_handle = eval_sess.run(eval_model.eval_iterator.string_handle())
    dev_ppl, _ = run_internal_eval(eval_model, eval_handle, eval_sess,
            hparams.out_dir, hparams, summary_writer)
    utils.add_summary(summary_writer, global_step, "dev_ppl", dev_ppl)
    total_inference_time = time.time() - start_time
    utils.add_summary(summary_writer, global_step, "infer_time",
            total_inference_time)


def infer_fn(hparams, identity, scope=None, extra_args=None, target_session=""):
  """main entry point for inference and evaluation."""
  # create infer and eval models
  infer_model = model_helper.create_infer_model(
      diag_model.Model, hparams, scope, extra_args=extra_args)
  eval_model = model_helper.create_eval_model(diag_model.Model, hparams, scope)
  config_proto = utils.get_config_proto(
      log_device_placement=hparams.log_device_placement,
      allow_soft_placement=True)
  # create the eval session
  eval_sess = tf.Session(
      target=target_session, config=config_proto, graph=eval_model.graph)

  secondary_fn_tmp(hparams, identity, hparams.out_dir, infer_model, eval_model,
                   eval_sess, "infer", single_worker_inference)


def secondary_fn_tmp(hparams, identity, model_dir, model, eval_model, eval_sess,
                     name, worker_fn):
  """secondary helper function for inference and evaluation."""
  steps_per_external_eval = 10
  # initialize summary writer
  if not hparams.codalab:
    summary_writer_path = os.path.join(hparams.out_dir, identity + name + "_log")
    print("summary_writer_path", summary_writer_path)
    summary_writer = tf.summary.FileWriter(summary_writer_path, model.graph)
  else:
    summary_writer = None
  config_proto = utils.get_config_proto(
      log_device_placement=hparams.log_device_placement,
      allow_soft_placement=True)
  # create session
  sess = tf.Session(config=config_proto, graph=model.graph)

  # wait for the checkpoints
  latest_ckpt = None
  last_external_eval_step = 0

  # main inference loop
  while True:
    latest_ckpt = tf.contrib.training.wait_for_new_checkpoint(
        model_dir, latest_ckpt)
    with model.graph.as_default():
      _, global_step = model_helper.create_or_load_model(
          model.model, model_dir, sess, name)
    if global_step - last_external_eval_step >= steps_per_external_eval:
      last_external_eval_step = global_step
      worker_fn(model, sess, eval_model, eval_sess, latest_ckpt, summary_writer,
                global_step, hparams)
    if not hparams.eval_forever:
      break  # if eval_foever is disabled, we only evaluate once
  if summary_writer:
    summary_writer.close()
  sess.close()
