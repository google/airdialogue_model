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

"""To perform inference on test set given a trained model."""

import copy
import os
import random
import re
import time
import json
from tqdm import tqdm
import math
import numpy as np
import tensorflow as tf
import model as diag_model
import model_helper
from dialogue import SelfplayDialogue
from utils import dialogue_utils
from utils import misc_utils as utils
from utils.dialogue_utils import task_SP_DISTRIBUTED


def handle_summary(diag_mode, summary_writer, global_step, all_summary,
                   summary_weight):
  """hanel all summary and combine them together."""
  combined = {}
  for summary in all_summary:
    for key in summary:
      if key not in combined:
        combined[key] = []
      combined[key].append(summary[key])
  print('combined', combined)
  for key in combined:
    combined[key] = np.average(combined[key], weights=summary_weight)
    name = diag_mode + '_' + key
    utils.add_summary(summary_writer, global_step, name, combined[key])

def pred_action_to_obj(pred_action):
    action_obj = {
        'name': ' '.join([pred_action[0], pred_action[1]]),
        'flight': [''],
        'status': ''
    }
    fl_match = re.match('<fl_(\d+)>', pred_action[2])
    if fl_match:
        action_obj['flight'][0] = fl_match[0]
    status_match = re.match('<st_(\w+)>', pred_action[3])
    if status_match:
        action_obj['status'] = status_match[0]
    return action_obj

def utterance_to_dialogue(utt):
    stack = ""
    dialogue = []
    for s in utt:
        if s == "<t1>" or s == "<t2>":
            if stack:
                dialogue.append(stack)
                stack = ""
            stack += "customer:" if s == "<t1>" else "agent:"
        elif s == "<eod>":
            break
        else:
            stack += " " + s
    if stack:
        dialogue.append(stack)
    return dialogue

def output_generated_data(generated_data, eval_out):
  bs_intent, bs_pred_action, bs_truth_action, utt_arr, bs_kb = generated_data
  for intent, pred_action, true_action, utterance, kb in zip(
      bs_intent, bs_pred_action, bs_truth_action, utt_arr, bs_kb):

    generated_obj = {
        # 'intent': intent,
        'pred_action': pred_action_to_obj(pred_action),
        # 'action': true_action,
        'dialogue': utterance_to_dialogue(utterance),
        # 'kb': kb
    }
    # print('generated_obj', generated_obj)
    eval_out.write(json.dumps(generated_obj) + '\n')


def single_worker_selfplay(mutable_model, immutable_model, mutable_sess,
                           immutable_sess, selfplay_data_file, selfplay_kb_file,
                           global_step, hparams, summary_writer):
  """selfplay with a single worker.

  This is preminarily used for self play
  evaluation.
  """

  dialogue_mode = dialogue_utils.mode_self_play_dialogue_eval
  # Read self play data
  selfplay_data = dialogue_utils.load_data(selfplay_data_file)
  selfplay_kb = dialogue_utils.load_data(selfplay_kb_file)

  # construct dialogue object
  dialogue = SelfplayDialogue(
      mutable_model,
      immutable_model,
      mutable_sess,
      immutable_sess,
      hparams.max_dialogue_turns,
      hparams.train_threadhold,
      hparams.start_of_turn1,
      hparams.start_of_turn2,
      hparams.end_of_dialogue,
      summary_writer=summary_writer,
      dialogue_mode=dialogue_mode,
      hparams=hparams)

  batch_size = dialogue.self_play_eval_batch_size
  assert batch_size <= len(selfplay_data)

  loaded_mutable, _ = load_self_play_model(
      dialogue.mutable_model, dialogue.mutable_sess, 'mutable',
      hparams.self_play_pretrain_dir, hparams.out_dir)
  loaded_immutable, _ = load_self_play_model(
      dialogue.immutable_model, dialogue.immutable_sess, 'immutable',
      hparams.self_play_pretrain_dir, hparams.out_dir)
  worker_step = 0
  all_summary = []
  summary_weight = []  # used in combination with all_summary

  # max_eval_per_flip = 100000
  # We flip the role of the agent for exactly two times. In the first iteration
  # when flip = 0, mutable model will be agent 1 and immutable model will be
  # agent 2. The other way around when flip = 1.
  start_time = time.time()
  num_flips_for_initial_speaker = 2
  with tf.gfile.GFile(hparams.selfplay_eval_output_file, 'w') as selfplay_out:
    print('flip 1')
    for flip in range(num_flips_for_initial_speaker):
      # epoch = -1
      i = len(selfplay_data)  # force shuffling at the beginning
      agent1, agent2, _ = dialogue.flip_agent(
          (loaded_mutable, mutable_sess, dialogue.mutable_handles),
          (loaded_immutable, immutable_sess, dialogue.immutable_handles), flip)
      # only eval one epoch
      # while epoch <= 0:
        # print(i, max_eval_per_flip)
      # if i * batch_size >= len(selfplay_data):  # reacehd the end
      input_data = list(zip(selfplay_data, selfplay_kb))
      # we don't shuffle in evaluation
      # random.shuffle(input_data)  # random shuffle input data
      # i = 0
      selfplay_data, selfplay_kb = list(zip(*input_data))
      # epoch += 1
      ceil = int(math.ceil(len(selfplay_data) *1.0 / batch_size))
      for i in tqdm(list(range(0, ceil))):
        start_ind = i * batch_size
        end_ind = min(i * batch_size + batch_size, len(selfplay_data))

        batch_data = selfplay_data[start_ind:end_ind]
        batch_kb = selfplay_kb[start_ind:end_ind]
        # we indicate to let agent1 to talk first. Keep in mind that we will
        # swap between agent1 and agent2.
        speaker = flip % 2
        generated_data, _, summary = dialogue.talk(hparams.max_dialogue_len,
                                                   batch_data, batch_kb, agent1,
                                                   agent2, worker_step,
                                                   end_ind - start_ind, speaker)
        output_generated_data(generated_data, selfplay_out)
        all_summary.append(summary)
        # number of elements processed
        summary_weight.append(end_ind - start_ind)
        worker_step += 1
  handle_summary(dialogue_mode, summary_writer, global_step, all_summary,
                 summary_weight)
  end_time = time.time()
  print('finished')
  utils.add_summary(summary_writer, global_step, dialogue_mode + '_time',
                    end_time - start_time)  #  step wise summary


def load_self_play_model(model, sess, identity, supervised_learning_path,
                         self_play_path):
  """This function loads the self-play model.

  It will first check the self play
  directory. If it's empty it will then load the pre-trained model from
  supervised learning.
  """
  ckpt = tf.train.latest_checkpoint(self_play_path)
  # first try self_play out dir
  if ckpt:
    print('{0} restore from self_play path at {1}'.format(
        identity, self_play_path))
    with model.graph.as_default():
      model_helper.full_restore(sess, ckpt)
  # if model doesn't exist then load supervised learning model
  else:
    print('{0} restore from supervised learning at {1}'.format(
        identity, supervised_learning_path))
    ckpt = tf.train.latest_checkpoint(supervised_learning_path)
    assert ckpt
    with model.graph.as_default():
      # first do initialization to make sure that all variables are initialized
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      model_helper.full_restore(sess, ckpt)
  return model, sess


def self_play_eval_fn(hparams,
                      identity,
                      num_workers=1,
                      jobid=0,
                      scope=None,
                      target_session=''):
  """This is the single worker self play.

  Mostly used for self play
  evaluation. identity is used here to distinguish between workers.
  """
  model_creator = diag_model.Model

  mutable_model = model_helper.create_selfplay_model(
      model_creator,
      True,  # mutable is True
      num_workers,
      jobid,
      hparams=hparams,
      scope=scope)
  immutable_model = model_helper.create_selfplay_model(
      model_creator,
      False,  # mutable is False
      num_workers,
      jobid,
      hparams=hparams,
      scope=scope)

  mutable_sess = tf.Session(
      graph=mutable_model.graph,
      config=tf.ConfigProto(
          allow_soft_placement=True, device_count={'GPU': hparams.num_gpus}))
  immutable_sess = tf.Session(
      graph=immutable_model.graph,
      config=tf.ConfigProto(
          allow_soft_placement=True, device_count={'GPU': hparams.num_gpus}))

  # number of steps per external eval
  steps_per_external_eval = 10
  # force conducting a self play at the beginning
  last_external_eval_step = -1 * steps_per_external_eval
  print('hparams.self_play_pretrain_dir=', hparams.self_play_pretrain_dir)
  print('steps_per_external_eval=', steps_per_external_eval)

  writer_path = os.path.join(hparams.out_dir,
                             identity + hparams.task_type + '_log')
  summary_writer = tf.summary.FileWriter(writer_path, mutable_sess.graph)
  print('summary_writer estabilished at', writer_path)

  # waiting for checkpoints and loop forever
  latest_ckpt = None
  while True:
    latest_ckpt = tf.contrib.training.wait_for_new_checkpoint(
        hparams.out_dir, latest_ckpt)
    print('got checkpoint', latest_ckpt)
    # get the global_step variable first
    with mutable_model.graph.as_default():
      # first initialize to avoid encountering missing component for adam optimizer
      _, global_step = model_helper.create_or_load_model(
          mutable_model.model, hparams.out_dir, mutable_sess, hparams.task_type)
    # valid evaluation step
    if (not hparams.eval_forever) or (global_step - last_external_eval_step >=
                                      steps_per_external_eval):
      # if eval_forever is disabled, we will do one selfplay evalation
      # otherwise, we will wait until certain number of timesteps are elapsed.
      last_external_eval_step = global_step
      print('do single worker evaluation')
      single_worker_selfplay(mutable_model, immutable_model, mutable_sess,
                             immutable_sess, hparams.self_play_eval_data,
                             hparams.self_play_eval_kb, global_step, hparams,
                             summary_writer)
    else:
      print('Wait until steps_per_external_eval is reached.', global_step,
            last_external_eval_step, steps_per_external_eval)
    if not hparams.eval_forever:
      break  # if eval_foever is disabled, we only evaluate once

  mutable_sess.close()
  immutable_sess.close()


def multi_worker_selfplay(hparams,
                          identity,
                          scope=None,
                          target_session='',
                          is_chief=True,
                          ps_tasks=0,
                          num_workers=1,
                          jobid=0,
                          startup_delay_steps=0):
  """This is the multi worker selfplay, mostly used for self play

  distributed training.
  identity is used.
  """
  immutable_model_reload_freq = hparams.immutable_model_reload_freq
  # 1. models and summary writer
  model_creator = diag_model.Model
  extra_args = model_helper.ExtraArgs(
      single_cell_fn=None,
      model_device_fn=tf.train.replica_device_setter(ps_tasks),
      attention_mechanism_fn=None)

  mutable_model = model_helper.create_selfplay_model(
      model_creator,
      is_mutable=True,
      num_workers=num_workers,
      jobid=jobid,
      hparams=hparams,
      scope=scope,
      extra_args=extra_args)
  immutable_hparams = copy.deepcopy(hparams)
  immutable_hparams.num_gpus = 0
  immutable_model = model_helper.create_selfplay_model(
      model_creator,
      is_mutable=False,
      num_workers=num_workers,
      jobid=jobid,
      hparams=immutable_hparams,
      scope=scope)

  if hparams.self_play_immutable_gpu:
    print('using GPU for immutable')
    immutable_sess = tf.Session(
        graph=immutable_model.graph,
        config=tf.ConfigProto(allow_soft_placement=True))
  else:
    print('not using GPU for immutable')
    immutable_sess = tf.Session(
        graph=immutable_model.graph,
        config=tf.ConfigProto(
            allow_soft_placement=True, device_count={'GPU': 0}))

  immutable_model, immutable_sess = load_self_play_model(
      immutable_model, immutable_sess, 'immutable',
      hparams.self_play_pretrain_dir, hparams.out_dir)
  global_step = immutable_model.model.global_step.eval(session=immutable_sess)

  if is_chief:
    ckpt = tf.train.latest_checkpoint(hparams.out_dir)
    if not ckpt:
      print('global_step, saving pretrain model to hparams.out_dir',
            global_step, hparams.out_dir)
      immutable_model.model.saver.save(  # this is the prevent adam error
          immutable_sess,
          os.path.join(hparams.out_dir, 'dialogue.ckpt'),
          global_step=global_step)
      print('save finished')

  if is_chief:
    summary_writer_path = os.path.join(hparams.out_dir,
                                       identity + task_SP_DISTRIBUTED + '_log')
    summary_writer = tf.summary.FileWriter(summary_writer_path,
                                           mutable_model.graph)
    print('summary writer established at', summary_writer_path)
  else:
    summary_writer = None
  # 2. supervisor and sessions

  sv = tf.train.Supervisor(
      graph=mutable_model.graph,
      is_chief=is_chief,
      saver=mutable_model.model.saver,
      save_model_secs=0,  # disable automatic save checkpoints
      summary_op=None,
      logdir=hparams.out_dir,
      checkpoint_basename='dialogue.ckpt')

  mutable_config = utils.get_config_proto(
      log_device_placement=hparams.log_device_placement,
      allow_soft_placement=True)
  mutable_config.device_count['GPU'] = hparams.num_gpus

  mutable_sess = sv.prepare_or_wait_for_session(
      target_session,
      config=mutable_config)

  # 3. additiona preparations
  global_step = mutable_model.model.global_step.eval(session=mutable_sess)
  while global_step < (jobid * (jobid + 1) * startup_delay_steps / 2):
    time.sleep(1)
    global_step = mutable_model.model.global_step.eval(session=mutable_sess)

  # save first model
  if is_chief:
    print('saving the first checkpoint to', hparams.out_dir)
    mutable_model.model.saver.save(
        mutable_sess,
        os.path.join(hparams.out_dir, 'dialogue.ckpt'),
        global_step=global_step)
    last_save_step = global_step

  # Read data
  selfplay_data = dialogue_utils.load_data(hparams.self_play_train_data)
  selfplay_kb = dialogue_utils.load_data(hparams.self_play_train_kb)

  dialogue = SelfplayDialogue(
      mutable_model,
      immutable_model,
      mutable_sess,
      immutable_sess,
      hparams.max_dialogue_turns,
      hparams.train_threadhold,
      hparams.start_of_turn1,
      hparams.start_of_turn2,
      hparams.end_of_dialogue,
      summary_writer=summary_writer,
      dialogue_mode=task_SP_DISTRIBUTED,
      hparams=hparams)

  # 4. main loop
  last_immmutable_model_reload = global_step
  last_save_step = global_step
  batch_size = dialogue.batch_size
  assert batch_size <= len(selfplay_data)

  # this is the start point of the self-play data. force shuffling at the beginning
  i = len(selfplay_data)
  train_stats = [0, 0]
  while global_step < hparams.num_self_play_train_steps + hparams.num_train_steps:
    # a. reload immutable model, muttable will be automated managed by supervisor
    if immutable_model_reload_freq > 0 and global_step - last_immmutable_model_reload > immutable_model_reload_freq:
      immutable_model, immutable_sess = load_self_play_model(
          immutable_model, immutable_sess, 'immutable',
          hparams.self_play_pretrain_dir, hparams.out_dir)
      last_immmutable_model_reload = global_step
    # b. possiblely flip between speakers (or roll out models),
    # based on either a random policy or by step counts
    agent1, agent2, mutable_agent_index = dialogue.flip_agent(
        (mutable_model, mutable_sess, dialogue.mutable_handles),
        (immutable_model, immutable_sess, dialogue.immutable_handles))
    train_stats[mutable_agent_index] += 1
    # read selfplay data
    start_time = time.time()
    if i * batch_size + batch_size > len(selfplay_data):  # reached the end
      input_data = list(zip(selfplay_data, selfplay_kb))
      random.shuffle(input_data)  # random shuffle input data
      i = 0
      selfplay_data, selfplay_kb = list(zip(*input_data))

    start_ind, end_ind = i * batch_size, i * batch_size + batch_size
    batch_data, batch_kb = selfplay_data[start_ind:end_ind], selfplay_kb[
        start_ind:end_ind]
    train_example, _, _ = dialogue.talk(hparams.max_dialogue_len, batch_data,
                                        batch_kb, agent1, agent2, global_step,
                                        batch_size)
    possible_global_step = dialogue.maybe_train(
        train_example, mutable_agent_index, global_step, force=True)
    if possible_global_step:
      global_step = possible_global_step
    if is_chief and global_step - last_save_step > hparams.self_play_dist_save_freq:
      mutable_model.model.saver.save(
          mutable_sess,
          os.path.join(hparams.out_dir, 'dialogue.ckpt'),
          global_step=global_step)
      last_save_step = global_step
    end_time = time.time()

    if is_chief:
      utils.add_summary(summary_writer, global_step,
                        task_SP_DISTRIBUTED + '_' + 'time',
                        end_time - start_time)
      utils.add_summary(summary_writer, global_step,
                        task_SP_DISTRIBUTED + '_' + 'train_ratio',
                        train_stats[0] * 1.0 / (train_stats[1] + 0.1))
    i += 1

  if is_chief:
    summary_writer.close()

  mutable_sess.close()
  immutable_sess.close()
