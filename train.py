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

"""train the dialogue model."""

import math
import os
import time
import tensorflow.compat.v1 as tf
import model as diag_model
import model_helper
from utils import misc_utils as utils


def train(hparams, identity, scope=None, target_session=""):
  """main loop to train the dialogue model. identity is used."""
  out_dir = hparams.out_dir
  steps_per_stats = hparams.steps_per_stats
  steps_per_internal_eval = 3 * steps_per_stats

  model_creator = diag_model.Model

  train_model = model_helper.create_train_model(model_creator, hparams, scope)

  model_dir = hparams.out_dir

  # Log and output files
  log_file = os.path.join(out_dir, identity+"log_%d" % time.time())
  log_f = tf.gfile.GFile(log_file, mode="a")
  utils.print_out("# log_file=%s" % log_file, log_f)

  avg_step_time = 0.0

  # load TensorFlow session and model
  config_proto = utils.get_config_proto(
      log_device_placement=hparams.log_device_placement,
      allow_soft_placement=True)

  train_sess = tf.Session(
      target=target_session, config=config_proto, graph=train_model.graph)

  train_handle = train_sess.run(train_model.train_iterator.string_handle())

  with train_model.graph.as_default():
    loaded_train_model, global_step = model_helper.create_or_load_model(
        train_model.model, model_dir, train_sess, "train")

  # initialize summary writer
  summary_writer = tf.summary.FileWriter(
      os.path.join(out_dir, "train_log"), train_model.graph)

  last_stats_step = global_step
  last_eval_step = global_step

  # initialize training stats.
  step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
  checkpoint_total_count = 0.0
  speed, train_ppl = 0.0, 0.0
  start_train_time = time.time()

  utils.print_out(
      "# Start step %d, lr %g, %s" %
      (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
       time.ctime()),
      log_f)

  # initialize iterators
  skip_count = hparams.batch_size * hparams.epoch_step
  utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
  train_sess.run(
      train_model.train_iterator.initializer,
      feed_dict={train_model.skip_count_placeholder: skip_count})

  # main training loop
  while global_step < hparams.num_train_steps:
    start_time = time.time()
    try:  #  run a step
      step_result = loaded_train_model.train(train_sess, train_handle)
      (_, step_loss, all_summaries, step_predict_count, step_summary,
       global_step, step_word_count, batch_size, _, _, words1, words2, mask1,
       mask2) = step_result
      hparams.epoch_step += 1

    except tf.errors.OutOfRangeError:  # finished an epoch
      hparams.epoch_step = 0
      utils.print_out("# Finished an epoch, step %d." % global_step)
      train_sess.run(
          train_model.train_iterator.initializer,
          feed_dict={train_model.skip_count_placeholder: 0})
      continue

    # Write step summary.
    summary_writer.add_summary(step_summary, global_step)
    for key in all_summaries:
      utils.add_summary(summary_writer, global_step, key, all_summaries[key])

    # update statistics
    step_time += (time.time() - start_time)

    checkpoint_loss += (step_loss * batch_size)
    checkpoint_predict_count += step_predict_count
    checkpoint_total_count += float(step_word_count)

    if global_step - last_stats_step >= steps_per_stats:
      # print statistics for the previous epoch and save the model.
      last_stats_step = global_step

      avg_step_time = step_time / steps_per_stats
      utils.add_summary(summary_writer, global_step, "step_time", avg_step_time)
      train_ppl = utils.safe_exp(checkpoint_loss / checkpoint_predict_count)
      speed = checkpoint_total_count / (1000 * step_time)
      if math.isnan(train_ppl):
        break

      # Reset timer and loss.
      step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
      checkpoint_total_count = 0.0

      # save the model
      loaded_train_model.saver.save(
          train_sess,
          os.path.join(out_dir, "dialogue.ckpt"),
          global_step=global_step)

      # print the dialogue if in debug mode
      if hparams.debug:
        utils.print_current_dialogue(words1, words2, mask1, mask2)

    #  write out internal evaluation
    if global_step - last_eval_step >= steps_per_internal_eval:
      last_eval_step = global_step

      utils.print_out("# Internal Evaluation. global step %d" % global_step)
      utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

  # finished training
  loaded_train_model.saver.save(
      train_sess,
      os.path.join(out_dir, "dialogue.ckpt"),
      global_step=global_step)
  result_summary = ""
  utils.print_out(
      "# Final, step %d lr %g "
      "step-time %.2f wps %.2fK ppl %.2f, %s, %s" %
      (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
       avg_step_time, speed, train_ppl, result_summary, time.ctime()),
      log_f)
  utils.print_time("# Done training!", start_train_time)
  utils.print_out("# Start evaluating saved best models.")
  summary_writer.close()
