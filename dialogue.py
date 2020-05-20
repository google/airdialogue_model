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

"""The conversation module used for self play."""


import math
import sys
import numpy as np
import tensorflow as tf
from utils import dialogue_utils
from utils import misc_utils as utils


class Conversation(object):
  """The Conversation class models the behavior of a single self-play conversation.
  """

  def get_initial_utterance(self, speaker):
    # the situation is different here because we removed a speaker flip right
    # aftet this is initialized
    if speaker == 0:
      return ['<t1>']  # it means we let speaker 0 to talk now
    else:
      return ['<t2>']

  def __init__(self, max_diag_len, turn1_token, turn2_token, num_utterance,
               speaker):
    self.utt_arr = []
    self.action_arr = []
    self.is_finished = []
    self.turn1_token = turn1_token
    self.turn2_token = turn2_token
    self.max_diag_len = max_diag_len

    for i in range(num_utterance):
      self.utt_arr.append([])
      self.is_finished.append(False)
      self.action_arr.append([])

    for i in range(num_utterance):
      utt = self.get_initial_utterance(speaker)
      self.utt_arr[i].extend(utt)

  def get_start_and_end_token(self, speaker):
    if speaker == 0:
      begin_token = self.turn1_token
      end_token = self.turn2_token
    else:
      begin_token = self.turn2_token
      end_token = self.turn1_token
    return begin_token, end_token

  def process(self, new_utterances1, new_utterances2, actions, speaker,
              last_round):

    def apply_filter(filter_token, utterance):
      try:
        ind = utterance.index(filter_token)
        # print ('ind found', ind,utterance[:ind] )
        return utterance[:ind], True
      except:  # filter_token not found
        return utterance, False

    begin_token, end_token = self.get_start_and_end_token(speaker)
    new_utterances_all = new_utterances1 if speaker == 0 else new_utterances2
    for i, (np_utterance, ac) in enumerate(zip(new_utterances_all, actions)):
      if self.is_finished[i]:
        continue
      new_utterances = []
      for token in np_utterance:
        new_utterances.append(token)
      new_utterances = list(map(lambda bs: bs.decode(), new_utterances))
      # 1. get sub_str before begin_tokens as they are invalid
      new_utterances, _ = apply_filter(begin_token, new_utterances)
      # 2. get sub_str before end token
      new_utterances, _ = apply_filter(end_token, new_utterances)
      # 3. get sub_str before end_of dialogue
      new_utterances, terminated = apply_filter('<eod>', new_utterances)
      # 4. cap on max_length
      remaining_informative_words = self.max_diag_len - len(self.utt_arr[i]) - 1
      if terminated:
        remaining_informative_words -= 1  # we need to add <eod>
      new_utterances = new_utterances[:remaining_informative_words]
      new_utterances = list(new_utterances)
      ##### start putting it together
      # 6. add eod
      if terminated:
        new_utterances.append('<eod>')

      if terminated or last_round:
        self.action_arr[i] = [s.decode() for s in ac]
        self.is_finished[i] = True
        # 7. add end token
      new_utterances.append(end_token)

      self.utt_arr[i].extend(new_utterances)

    return sum(self.is_finished) == len(self.is_finished)

  def get_train_data(self):
    # print("self.utt_arr", self.utt_arr)
    return self.utt_arr, self.action_arr


class SelfplayDialogue(object):
  """The SelfplayDialogue can be reused for multiple conversations."""

  def __init__(self, mutable_model, immutable_model, mutable_sess,
               immutable_sess, max_dialogue_turns, train_threadhold,
               turn1_token, turn2_token, eod_token, summary_writer,
               dialogue_mode, hparams):
    # model and session
    self.mutable_model = mutable_model
    self.immutable_model = immutable_model
    self.mutable_sess = mutable_sess
    self.immutable_sess = immutable_sess
    # iterators

    self.mutable_handles = self.mutable_sess.run([
        mutable_model.train_iterator.string_handle(),
        mutable_model.self_play_ft_iterator.string_handle(),
        mutable_model.self_play_st_iterator.string_handle()
    ])
    self.immutable_handles = self.immutable_sess.run([
        immutable_model.train_iterator.string_handle(),
        immutable_model.self_play_ft_iterator.string_handle(),
        immutable_model.self_play_st_iterator.string_handle()
    ])

    self.iterator_mode = 1  # 1 is fulltext, 2 is structured

    self.summary_writer = summary_writer
    self.dialogue_mode = dialogue_mode
    self.batch_size = hparams.self_play_batch_size
    self.self_play_eval_batch_size = hparams.self_play_eval_batch_size
    self.update_batch_size = hparams.self_play_update_batch_size
    self.hparams = hparams
    self.gamma = hparams.reward_discount

    assert mutable_model.model.mode == dialogue_utils.mode_self_play_mutable
    # parameters
    self.max_dialogue_turns = max_dialogue_turns

    # won't train the model until train_threadhold samples are reached
    self.train_threadhold = train_threadhold
    self.turn1_token = turn1_token
    self.turn2_token = turn2_token
    self.turn_tokens = [turn1_token, turn2_token]
    self.eod_token = eod_token
    # initialization
    self.train_samples = []
    self.train_counter = 0
    self.train_it_initialized = False
    ##### stats on rl vs sl updates
    self.num_rl_updates = 0
    self.num_sl_updates = 0

  def format_samples_batch(self,
                           batch_intent,
                           batch_pred_action,
                           batch_truth_action,
                           batch_utterance,
                           batch_reward_diag,
                           batch_reward_action,
                           batch_size,
                           boundary=None):
    output_data = []
    for i in range(batch_size):
      utterance = ' '.join(batch_utterance[i])
      if not boundary:
        boundary1 = self.get_dialogue_boundary(self.turn_tokens[0], utterance,
                                               self.turn_tokens[0],
                                               self.turn_tokens[1])
        boundary = boundary1[0] + boundary1[1]
      str_b = []
      for ele in boundary:
        str_b.append(str(ele))
      intent = batch_intent[i]
      pred_action = batch_pred_action[i]
      truth_action = batch_truth_action[i]
      reward_diag, reward_action = batch_reward_diag[i], batch_reward_action[i]
      arr = [
          intent, pred_action, truth_action, utterance, ' '.join(str_b),
          reward_diag, reward_action
      ]
      output_data.append('|'.join(arr))
    return output_data

  def generate_utterance_ordinary(self, data, kb, self_play_model, sess,
                                  batch_size, handles):

    if self.iterator_mode == 1:
      real_iterator = self_play_model.self_play_ft_iterator
    else:
      real_iterator = self_play_model.self_play_st_iterator

    sess.run(
        real_iterator.initializer,
        feed_dict={
            self_play_model.data_placeholder: data,
            self_play_model.kb_placeholder: kb,
            self_play_model.batch_size_placeholder: batch_size
        })

    iterator_handle = handles[self.iterator_mode]
    res = self_play_model.model.generate_self_play_utterance(
        sess, iterator_handle)
    return res

  def generate_utterance(self, batch_intent, conv, kb,
                         speaker, turn, batch_size):
    # preapre output
    utt = conv.get_train_data()
    composit_data = self.format_samples_batch(
        batch_intent=batch_intent,
        batch_pred_action=['s'] * batch_size,
        batch_truth_action=['s'] * batch_size,
        batch_utterance=utt[0],  # utterance
        batch_reward_diag=['0.5'] * batch_size,
        batch_reward_action=['0.5'] * batch_size,
        batch_size=batch_size)
    composit_kb = kb
    self_play_model, sess, handles = self.agents[speaker]
    new_utt1, new_utt2, new_action = self.generate_utterance_ordinary(
        composit_data, composit_kb, self_play_model, sess, batch_size, handles)
    all_finished = conv.process(new_utt1, new_utt2, new_action, speaker,
                                turn == self.max_dialogue_turns - 1)
    return all_finished

  def parse_input(self, batch_input_data, batch_input_kb):
    batch_intent = []
    batch_action = []
    batch_kb = batch_input_kb
    for input_data in batch_input_data:
      intent, action = input_data.split('|')
      batch_intent.append(intent)
      batch_action.append(action)

    return batch_intent, batch_action, batch_kb

  def do_rl_training(self, data, kb, batch_size, model, sess, speaker,
                     global_step, self_play_handle):
    if self.iterator_mode == 1:
      self_play_iterator = model.self_play_ft_iterator
    elif self.iterator_mode == 2:
      self_play_iterator = model.self_play_st_iterator
    else:
      raise Exception('not defined self_play_mode')

    # first do initialization
    sess.run(
        self_play_iterator.initializer,
        feed_dict={
            model.data_placeholder: data,
            model.kb_placeholder: kb,
            model.batch_size_placeholder: batch_size
        })
    # second, do training
    res = model.model.self_play(sess, speaker, self_play_handle)
    all_summaries = res[-1]
    if self.summary_writer:
      for key in all_summaries:
        utils.add_summary(self.summary_writer, global_step,
                          self.dialogue_mode + '_' + key, all_summaries[key])
    global_step = res[2]
    self.num_rl_updates += 1
    return global_step

  def do_SL_training(self, model, sess, global_step, train_handle):
    # first do initialization
    if not self.train_it_initialized:
      sess.run(
          model.train_iterator.initializer,
          feed_dict={model.skip_count_placeholder: 0})
      self.train_it_initialized = True

    # second, do training
    while True:  # keep tring until no exception
      try:
        res = model.model.self_play_train(sess, train_handle)
        break
      except tf.errors.OutOfRangeError:
        sess.run(
            model.train_iterator.initializer,
            feed_dict={model.skip_count_placeholder: 0})
        continue

    all_summaries = res[-1]
    if self.summary_writer:
      for key in all_summaries:
        utils.add_summary(self.summary_writer, global_step,
                          self.dialogue_mode + '_' + key, all_summaries[key])
    global_step = res[-2]
    self.num_sl_updates += 1
    return global_step

  def get_dialogue_boundary(self, start_token, flat_dialogue, start_of_turn1,
                            start_of_turn2):

    def get_end_token(start, set_of_end_tokens, splitted_dialogues):
      for i in range(start, len(splitted_dialogues)):
        if splitted_dialogues[i] in set_of_end_tokens:
          return i
      assert False, 'end token not found :' + ' start=' + str(
          start) + '/' + str(len(splitted_dialogues))

    def get_next_start_token(end_position, start_token, splitted_dialogues):
      for i in range(end_position, len(splitted_dialogues)):
        if splitted_dialogues[i] == start_token:
          return i
      return len(splitted_dialogues)

    set_of_end_tokens = set([
        start_of_turn1, start_of_turn2
    ])  # taking out end_of_dialogue token because of dynamic rnn decoder
    splitted_dialogue = flat_dialogue.split(' ')
    i = get_next_start_token(0, start_token, splitted_dialogue)
    all_starts = []
    all_ends = []
    while i < len(splitted_dialogue
                 ) - 1:  # we don't find the end token for the last turn change.
      end_position = get_end_token(i + 1, set_of_end_tokens, splitted_dialogue)
      assert splitted_dialogue[end_position] != start_token, (
          'start token '
          'appeared twice') + ''.join(flat_dialogue)
      all_starts.append(i)
      all_ends.append(end_position)
      i = get_next_start_token(i + 1, start_token, splitted_dialogue)
    return (all_starts, all_ends)


  def scale_reward_batch(self, b_final_reward, gamma, b_diag):
    batch_reward_diag = []
    batch_reward_action = []
    for final_reward, diag in zip(b_final_reward, b_diag):
      diag_len = len(diag)
      reward_len = diag_len + self.hparams.len_action
      all_ind = list(range(reward_len - 1, -1, -1))
      all_rewards = []
      for i in range(len(all_ind)):
        all_rewards.append(str(math.pow(gamma, all_ind[i]) * final_reward))
      reward_diag = all_rewards[0:-1 * self.hparams.len_action]
      reward_action = all_rewards[-1 * self.hparams.len_action:]
      batch_reward_diag.append(' '.join(reward_diag))
      batch_reward_action.append(' '.join(reward_action))
    return batch_reward_diag, batch_reward_action


  def maybe_train(self, sample, speaker, global_step, force=False):
    self.train_samples.append(sample)
    if force or len(self.train_samples) >= self.train_threadhold:
      # first generate training examples
      data_arr = []
      kb_arr = []
      for sample in self.train_samples:  # each sample is a batch of data
        intent, pred_action, truth_action, utterance, kb = sample  # batch version
        all_rewards = dialogue_utils.compute_reward_batch(
            utterance, pred_action, truth_action, kb,
            self.hparams)  # batch version
        train_reward, _, _, _, _, _, _, _, _ = all_rewards
        final_reward = train_reward
        reward_diag, reward_action = self.scale_reward_batch(
            final_reward, self.gamma, utterance)  # in batches
        flat_pred_action = []
        for k in range(len(pred_action)):
          flat_pred_action.append(' '.join(pred_action[k]))

        new_data_arr = self.format_samples_batch(
            batch_intent=intent,
            batch_pred_action=flat_pred_action,
            batch_truth_action=truth_action,
            batch_utterance=utterance,
            batch_reward_diag=reward_diag,
            batch_reward_action=reward_action,
            batch_size=self.update_batch_size)
        data_arr.extend(new_data_arr)
        kb_arr.extend(kb)
      data_output, kb_output = data_arr, kb_arr
      new_global_step = None
      self.train_samples = []  # clean up
      self_play_hangle = self.mutable_handles[self.iterator_mode]
      if self.hparams.rl_training:
        new_global_step = self.do_rl_training(
            data_output, kb_output, self.update_batch_size, self.mutable_model,
            self.mutable_sess, speaker, global_step, self_play_hangle)

      print('self.hparams.self_play_sl_multiplier=',
            self.hparams.self_play_sl_multiplier)
      if self.hparams.self_play_sl_multiplier >= 0:  # train multiple or don't train at all
        print('do', self.hparams.self_play_sl_multiplier, 'supervised training')
        for _ in range(self.hparams.self_play_sl_multiplier):
          new_global_step = self.do_SL_training(self.mutable_model,
                                                self.mutable_sess, global_step,
                                                self.mutable_handles[0])
      else:
        print('do one supervised traiing')
        if self.train_counter >= abs(self.hparams.self_play_sl_multiplier):
          new_global_step = self.do_SL_training(self.mutable_model,
                                                self.mutable_sess, global_step,
                                                self.mutable_handles[0])
          self.train_counter = 0
        else:
          self.train_counter += 1

      if self.summary_writer:
        utils.add_summary(
            self.summary_writer, new_global_step,
            self.dialogue_mode + '_' + 'sl_rl',
            self.num_sl_updates * 1.0 / (self.num_rl_updates + 0.0001))

      return new_global_step
    return None

  def talk(self, max_diag_length, batch_input_data, batch_input_kb, agent1,
           agent2, worker_step, batch_size, speaker=None):
    """The main procedure to generate a single self play conversation."""
    # parse data
    bs_intent, bs_truth_action, bs_kb = self.parse_input(
        batch_input_data, batch_input_kb)
    # remember the roles of agents
    self.agents = [agent1, agent2]
    # In selfplay training the speaker will be non and we randomly chose an
    # initial speaker and initialize utterance.
    # In selfplay evaluation the speaker will be specified so we use as is
    if not speaker: speaker = int(np.random.random() < 0.5)
    # generate the conversation instance for this conversation.
    # print ('self.batch_size', self.batch_size)
    conv = Conversation(max_diag_length, self.turn1_token, self.turn2_token,
                        batch_size, speaker)

    # generate conversation by turn in batch mode until all conversations
    # terminated (finished = True) or the number of turns reached the maximum.
    turn = 0
    finished = False
    while (not finished) and turn < self.max_dialogue_turns:
      finished = self.generate_utterance(bs_intent, conv,
                                         bs_kb, speaker, turn, batch_size)
      #  Change the speaker as we move to the next turn.
      speaker = (speaker + 1) % 2
      turn += 1

    all_rewards = dialogue_utils.compute_reward_batch(
        conv.utt_arr, conv.action_arr, bs_truth_action, bs_kb, self.hparams)
    metrics = dialogue_utils.calculate_reward_metrics(all_rewards)
    metrics['num_turns'] = turn

    #  print out step stats only in debug mode
    if self.summary_writer and self.hparams.debug:
      for key in metrics:
        utils.add_summary(self.summary_writer, worker_step,
                          self.dialogue_mode + '_' + key + '_ws', metrics[key])

    utt_arr, bs_pred_action = conv.get_train_data()

    if self.hparams.debug:
      print('self_play debug: ' + bs_intent[0])
      print('self_play debug: all_rewards', all_rewards[0])
      print('self_play debug: ' + ' '.join(utt_arr[0]))
      print('self_play debug: ' + ' '.join(bs_pred_action[0]))
      sys.stdout.flush()
    return (bs_intent, bs_pred_action, bs_truth_action, utt_arr,
            bs_kb), turn, metrics

  def flip_agent(self, mutable_agent, immutable_agent, flip='random'):
    """This function flips the role of mutable agent and immutable agent so that

    they both have chances to play customer and agent. Remember both mutable
    immutable models actually contain two sub-models: customer and agent. We
    need to make sure that they have equal chances to serve as both parts when
    doing the self play. In self play evaluation, this is chosen
    deterministically based on the value of flip. In self play training, this
    is chosen randomly.
    """
    if flip == 'random':
      flip = int(np.random.random() < 0.5)
    if flip == 0:
      #  in the first flip mutable agent is agent 1 and immutable agent
      #  is agent 2.
      return mutable_agent, immutable_agent, flip
    else:
      #  in the second flip mutable agent is agent 2 and immutable agent
      #  is agent 1.
      return immutable_agent, mutable_agent, flip
