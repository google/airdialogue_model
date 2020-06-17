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
"""Utility functions specifically for airdialogue model."""

import codecs
import random
import re
import numpy as np
import tensorflow.compat.v1 as tf
from airdialogue.evaluator import infer_utils
from airdialogue.evaluator.metrics import f1
from airdialogue.evaluator.metrics.flight_distance import generate_scaled_flight
from airdialogue.evaluator.metrics.flight_distance import split_flight

from utils import misc_utils as utils
from utils import vocab_utils

mode_self_play_mutable = 'self_play_mutable'
mode_self_play_immutable = 'self_play_immutable'
self_play_modes = [
    mode_self_play_mutable,
    mode_self_play_immutable,
]
mode_self_play_dialogue_train = 'self_play_train'
mode_self_play_dialogue_eval = 'self_play_eval'

task_TRAINEVAL = 'TRAINEVAL'
task_INFER = 'INFER'
task_SP_EVAL = 'SP_EVAL'
task_SP_DISTRIBUTED = 'SP_DISTRIBUTED'


def compute_reward(predicted_action,
                   actual_action,
                   flight_db,
                   alpha=0.5,
                   beta=0.2,
                   gamma=0.3,
                   debug=False):
  """here we compute the scaled reward."""
  predicted_name, predicted_flight, predicted_state = parse_action(
      predicted_action)
  actual_name, actual_flight, actual_state = parse_action(actual_action)

  # this will do normalization including lower case and prouncation/space
  # removal
  score1 = f1.f1_score(predicted_name, actual_name)
  score2 = 1 - generate_scaled_flight(predicted_flight, actual_flight,
                                      flight_db)
  score3 = float(predicted_state == actual_state)

  reward_compliment = score1 * 0.2 + score2 * 0.5 + score3 * 0.3

  acc1 = score1
  acc2 = score2
  acc3 = score3
  return reward_compliment, acc1, acc2, acc3


def parse_action(action):
  """parse the action and consider multiple name scenario.

  name will also appear first.
  """
  name = ' '.join(action[0:-2])
  flight = action[-2]
  state = action[-1]
  return name, flight, state


def compute_01_score(predicted_action, actual_action):
  """here we compute the 0/1 score."""
  predicted_name, predicted_flight, predicted_state = parse_action(
      predicted_action)
  actual_name, actual_flight, actual_state = parse_action(actual_action)

  # name score discrete
  predicted_names = predicted_name.lower().split(' ')
  actual_names = actual_name.lower().split(' ')
  ds1_name = ((predicted_names[0].strip() == actual_names[0].strip()) +
              (predicted_names[1].strip() == actual_names[1].strip())) / 2.0

  # flight score discrete
  truth_idx_arr = split_flight(actual_flight)
  predicted_flight = predicted_flight.strip()
  if '<fl_empty>' in truth_idx_arr:
    assert len(truth_idx_arr) == 1
    ds2_flight = int(predicted_flight == '<fl_empty>')
  else:
    ds2_flight = int(predicted_flight in truth_idx_arr)
  # ds2_flight = predicted_flight.strip() == actual_flight.strip()

  # state score discrete
  ds3_state = predicted_state.strip() == actual_state.strip()

  # total score
  ds_total = 0.2 * ds1_name + 0.5 * ds2_flight + 0.3 * ds3_state

  return ds_total, ds1_name, ds2_flight, ds3_state


def get_training_reward(hparams, s1, s2, s3, d1, d2, d3):
  """Calcualte the reward for training."""
  if hparams.train_reward_type == 'scaled':
    return 0.2 * s1 + 0.5 * s2 + 0.3 * s3
  elif hparams.train_reward_type == 'discrete':
    return 0.2 * d1 + 0.5 * d2 + 0.3 * d3
  elif hparams.train_reward_type == 'combined':
    return 0.2 * (s1 + d1) / 2.0 + 0.5 * (s2 + d2) / 2.0 + 0.3 * (s2 + d3) / 2.0
  elif hparams.train_reward_type == 'extreme':
    return d1 * 1000 + d2 * 10 + d3 * 1
  else:
    raise ValueError('invalid reward type')


def compute_reward_batch(utterance,
                         predicted_action,
                         actual_action_concat,
                         flight_db,
                         hparams,
                         alpha=0.5,
                         beta=0.2,
                         gamma=0.3):
  """Calcualte the reward for a batch."""
  rewards = []
  acc1 = []
  acc2 = []
  acc3 = []
  discrete_score = []
  ds1_arr = []
  ds2_arr = []
  ds3_arr = []
  train_rw_arr = []
  for pa, aa_con, fl in zip(predicted_action, actual_action_concat, flight_db):
    aa = aa_con.split(' ')
    rw, ac1, ac2, ac3 = compute_reward(pa, aa, fl)
    rewards.append(rw)
    acc1.append(ac1)
    acc2.append(ac2)
    acc3.append(ac3)

    ds, ds1, ds2, ds3 = compute_01_score(pa, aa)
    discrete_score.append(ds)
    ds1_arr.append(ds1)
    ds2_arr.append(ds2)
    ds3_arr.append(ds3)
    train_rw_arr.append(
        get_training_reward(hparams, ac1, ac2, ac3, ds1, ds2, ds3))
  return train_rw_arr, rewards, acc1, acc2, acc3, discrete_score, ds1_arr, ds2_arr, ds3_arr


def calculate_reward_metrics(batch_rewards):
  train_rw_arr, rewards, acc1, acc2, acc3, discrete_score, ds1_arr, ds2_arr, ds3_arr = batch_rewards
  # print ('acc3',acc3,np.mean(acc3))
  return {
      'train_rw_arr': np.mean(train_rw_arr),
      'reawrds': np.mean(rewards),
      'rw1_name': np.mean(acc1),
      'rw2_flight_num': np.mean(acc2),
      'rw3_action_state': np.mean(acc3),
      'discrete_score': np.mean(discrete_score),
      'ds1_name': np.mean(ds1_arr),
      'ds2_flight': np.mean(ds2_arr),
      'ds3_state': np.mean(ds3_arr),
  }


def extract_best_beam_response(response):
  """Make sure outputs is of shape [batch_size, time]  when using beam search."""
  new_response = [
      extract_best_beam_single(response[0]),
      extract_best_beam_single(response[1]),
      extract_best_beam_single(response[2])
  ]
  return new_response


def extract_best_beam_single(sample_words):
  """Extract the best beam from the sampled words."""
  if sample_words.ndim == 3:  # if this is beam search
    # Original beam search output is in [batch_size, time, beam_width] shape.
    sample_words = sample_words.transpose([2, 0, 1])
    # After extraction it would be [batch_size, time]
    best_beam = sample_words[0, :, :]
    return best_beam
  else:
    return sample_words


def decode_and_evaluate(name,
                        model,
                        data_iterator_handle,
                        sess,
                        trans_file,
                        ref_file,
                        metrics,
                        hparams,
                        infer_src_data=None,
                        decode=True):
  """Decode a test set and compute a score according to the evaluation task."""
  # Decode
  cnt = 0
  last_cnt = 0
  if decode:
    with codecs.getwriter('utf-8')(tf.gfile.GFile(trans_file,
                                                  mode='wb')) as trans_f:
      trans_f.write('')  # Write empty string to ensure file is created.
      while True:
        try:
          ut1, ut2, action = model.generate_infer_utterance(
              sess, data_iterator_handle)
          batch_size = ut1.shape[0]
          for sent_id in range(batch_size):
            src = infer_src_data[cnt]
            speaker = get_speaker(src)
            nmt_outputs = [ut1, ut2][speaker]
            translation = get_translation_cut_both(nmt_outputs, sent_id,
                                                   hparams.t1.encode(),
                                                   hparams.t2.encode())
            translation = translation.decode('utf-8')
            if hparams.self_play_start_turn == 'agent':
              if '<eod>' in translation:
                ac_arr = [w.decode('utf-8') for w in action[sent_id]]
                name = ac_arr[0] + ' ' + ac_arr[1]
                flight = re.match(r'<fl_(\d+)>', ac_arr[2])
                flight = flight.group(1) if flight else ''
                status = re.match(r'<st_(\w+)>', ac_arr[3])
                status = status.group(1) if status else ''
                translation += '|' + '|'.join([name, flight, status])
              else:
                translation += '|||'
            trans_f.write(translation + '\n')
            cnt += 1
          if last_cnt - cnt >= 10000:  # 400k in total
            utils.print_out('cnt= ' + str(cnt))
            last_cnt += 10000
        except tf.errors.OutOfRangeError:
          break

  # Evaluation
  evaluation_scores = {}
  if ref_file and tf.gfile.Exists(trans_file):
    for metric in metrics:
      score = infer_utils.evaluate(ref_file, trans_file, metric)
      evaluation_scores[metric] = score
      utils.print_out('  %s %s: %.1f' % (metric, name, score))

  return evaluation_scores


def load_data(inference_input_file):
  """Load inference data.

  Note, dialogue context might contain multiple
  flights connected using underlines. e.g. flight1_flight2.
  """
  with codecs.getreader('utf-8')(tf.gfile.GFile(
      inference_input_file, mode='rb')) as f:
    text_data = f.read().splitlines()
  return text_data


def get_translation(nmt_outputs, sent_id, tgt_eos):
  """Given batch decoding outputs, select a sentence and turn to text."""
  # Select a sentence
  output = nmt_outputs[sent_id, :].tolist()
  if tgt_eos in output:
    output = output[:output.index(tgt_eos)]
  translation = utils.format_text(output)
  return translation


def get_translation_cut_both(nmt_outputs, sent_id, start_token, end_token):
  """Given batch decoding outputs, select a sentence and turn to text."""
  # Select a sentence
  output = nmt_outputs[sent_id, :].tolist()
  if start_token in output:
    output = output[:output.index(start_token)]
  if end_token in output:
    output = output[:output.index(end_token)]

  translation = utils.format_text(output)

  return translation


def get_speaker(text_data):
  turn_token = text_data.split('|')[-1].split(' ')[-1]
  if turn_token == vocab_utils.start_of_turn1:
    return 0
  if turn_token == vocab_utils.start_of_turn2:
    return 1
  raise ValueError('invalid ending for dialogue : ' + turn_token)


def _sample_decode(model, global_step, iterator_handle, sess, hparams,
                   real_iterator, sample_src_data, sample_tar_data, sample_kb,
                   iterator_src_placeholder, iterator_kb_placeholder,
                   iterator_batch_size_placeholder):
  """Pick a sentence and decode."""
  decode_id = random.randint(0, len(sample_src_data) - 1)
  utils.print_out('  # %d' % decode_id)
  speaker = get_speaker(sample_src_data[decode_id])
  iterator_feed_dict = {
      iterator_src_placeholder: [sample_src_data[decode_id]],
      iterator_kb_placeholder: [sample_kb[decode_id]],
      iterator_batch_size_placeholder: 1,
  }
  sess.run(real_iterator.initializer, feed_dict=iterator_feed_dict)

  nmt_outputs, _, source = model.generate_utterance(sess, speaker,
                                                    iterator_handle)

  if hparams.beam_width > 0:
    nmt_outputs = nmt_outputs[0]
  translation = get_translation(
      nmt_outputs,
      sent_id=0,  # there is only one sentence because batch size is 1
      tgt_eos=None)
  src = get_translation(
      source,
      sent_id=0,  # there is only one sentence because batch size is 1
      tgt_eos=None)
  src_dialogue = src
  utils.print_out('    src: %s' % src_dialogue)
  if sample_tar_data:
    tar_dialogue = sample_tar_data[decode_id].split('|')[-1]
    utils.print_out('    ref: %s' % tar_dialogue)
  utils.print_out('    ours: ' + str(translation) + ' (speaker' + str(speaker) +
                  ')')
