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
"""Contains iterators for training, evaluating and inference on the dialogue models.

There are two iterator generators: get_infer_iterator applies to
inference and self play tasks while get_iterator applies to supervised
training/evaluation tasks.
"""

import collections
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow import contrib


# len_action = 3
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "intent", "intent_len", "source",
                            "target", "dialogue_len", "action", "action_len",
                            "predicted_action", "reward_diag", "reward_action",
                            "kb", "has_reservation", "mask1", "mask2", "turns"))
):
  pass


def process_data(object_str, vocab_table):
  """prelinminary process of dialogue data."""
  separated = tf.string_split([object_str]).values
  indices = tf.cast(vocab_table.lookup(separated), tf.int32)
  return indices, tf.size(indices)


def process_kb(kb_tensor):
  """prelinminary process of knowledge base."""
  has_reservation = kb_tensor[0]
  kb_tensor = kb_tensor[1:]
  return has_reservation, kb_tensor


def process_entry_common(intent, action, dialogue, boundaries, kb, vocab_table,
                         t1_id, t2_id):
  """A common procedure to process each entry of the dialogue data."""

  def do_process_boundary(start_points, end_points, input_length, t1_id, t2_id,
                          all_tokenized_diag):
    """function that contains the majority of the logic to proess boundary."""
    masks_start = tf.sequence_mask(start_points, input_length)
    masks_end = tf.sequence_mask(end_points, input_length)
    xor_masks = tf.logical_xor(masks_start, masks_end)
    mask1 = tf.reduce_any(xor_masks, axis=0)
    mask2 = tf.logical_not(mask1)
    all_turn1 = tf.equal(all_tokenized_diag, t1_id)
    all_turn2 = tf.equal(all_tokenized_diag, t2_id)
    turn_point = tf.logical_or(all_turn1, all_turn2)
    turn_point = tf.cast(turn_point, dtype=tf.float32)
    return mask1, mask2, turn_point

  def process_boundary(boundaries, input_length, t1_id, t2_id, all_dialogue):
    """process the boundaries of the dialogue."""
    points = tf.string_split([boundaries]).values
    points_val = tf.string_to_number(points, out_type=tf.int32)
    siz = tf.size(points_val) // 2
    start_points, end_points = points_val[0:siz], points_val[siz:]
    return do_process_boundary(start_points, end_points, input_length, t1_id,
                               t2_id, all_dialogue)

  def process_dialogue(tensor_dialogue, size_dialogue, mask1, mask2,
                       turn_point):
    new_dialogue_size = size_dialogue - 1
    source = tensor_dialogue[0:-1]
    target = tensor_dialogue[1:]
    mask1 = mask1[0:-1]
    mask2 = mask2[0:-1]
    turn_point = turn_point[0:-1]
    return source, target, new_dialogue_size, mask1, mask2, turn_point

  tensor_intent, size_intent = process_data(intent, vocab_table)
  all_dialogue, size_dialogue = process_data(dialogue, vocab_table)
  tensor_action, size_action = process_data(action, vocab_table)
  tensor_kb, unused_size = process_data(kb, vocab_table)
  has_reservation, tensor_kb = process_kb(tensor_kb)
  mask1, mask2, turn_point = process_boundary(boundaries, size_dialogue, t1_id,
                                              t2_id, all_dialogue)
  source_diag, target_diag, size_dialogue, mask1, mask2, turn_point = process_dialogue(
      all_dialogue, size_dialogue, mask1, mask2, turn_point)
  return tensor_intent, size_intent, source_diag, target_diag, size_dialogue, tensor_action, size_action, tensor_kb, has_reservation, mask1, mask2, turn_point


def process_entry_supervised(intent, action, dialogue, boundaries, kb,
                             vocab_table, t1_id, t2_id):
  """Pre-process procedure for the supervised iterator."""
  res = process_entry_common(intent, action, dialogue, boundaries, kb,
                             vocab_table, t1_id, t2_id)
  tensor_intent, size_intent, source_diag, target_diag, size_dialogue, tensor_action, size_action, tensor_kb, has_reservation, mask1, mask2, turn_point = res
  return tensor_intent, size_intent, source_diag, target_diag, size_dialogue, tensor_action, size_action, tf.constant(
      [0]), tf.constant([0.0]), tf.constant(
          [0.0]), tensor_kb, has_reservation, mask1, mask2, turn_point


def process_entry_self_play(intent, action, truth_action, kb, utterance,
                            boundary, reward_diag, reward_action, vocab_table):
  """Pro-proess procedure for the self-play iterator."""
  t1_id = tf.cast(vocab_table.lookup(tf.constant("<t1>")), tf.int32)
  t2_id = tf.cast(vocab_table.lookup(tf.constant("<t2>")), tf.int32)
  res = process_entry_common(intent, action, utterance, boundary, kb,
                             vocab_table, t1_id, t2_id)
  tensor_intent, size_intent, source_diag, target_diag, size_dialogue, tensor_action, size_action, tensor_kb, has_reservation, mask1, mask2, turn_point = res
  truth_action, _ = process_data(truth_action, vocab_table)
  splitted_reward_d = tf.string_split([reward_diag]).values
  splitted_reward_a = tf.string_split([reward_action]).values

  tensor_reward_diag = tf.string_to_number(
      splitted_reward_d, out_type=tf.float32,
      name=None)[:-1]  # remove the last dialogue ???
  tensor_reward_action = tf.string_to_number(
      splitted_reward_a, out_type=tf.float32, name=None)
  return tensor_intent, size_intent, source_diag, target_diag, size_dialogue, tensor_action, size_action, truth_action, tensor_reward_diag, tensor_reward_action, tensor_kb, has_reservation, mask1, mask2, turn_point


def process_entry_infer(intent, dialogue_context, kb, vocab_table):
  """pre-process procedure for inference iterator."""
  tensor_intent, size_intent = process_data(intent, vocab_table)
  tensor_diag, size_diag = process_data(dialogue_context, vocab_table)
  tensor_kb, unused_size = process_data(kb, vocab_table)
  has_reservation, tensor_kb = process_kb(tensor_kb)
  tensor_diag = tensor_diag[0:-1]  # delete last token,
  size_diag = size_diag - 1
  return tensor_intent, size_intent, tensor_diag, tf.constant(
      [0]), size_diag, tf.constant([0]), tf.constant(1), tf.constant(
          [0]), tf.constant([0.0]), tf.constant(
              [0.0]), tensor_kb, has_reservation, tf.constant(
                  [False]), tf.constant([False]), tf.constant([0.0])


def get_sub_items_supervised(data, kb):
  """process procedure for supervised learning."""
  all_data = tf.string_split([data], sep="|").values
  intent, action, dialogue, boundaries = all_data[0], all_data[1], all_data[
      2], all_data[3]
  return intent, action, dialogue, boundaries, kb


def get_sub_items_infer(data, kb):
  """process procedure for inference."""
  all_data = tf.string_split([data], sep="|", skip_empty=False).values
  intent, dialogue_context = all_data[0], all_data[1]
  return intent, dialogue_context, kb


def get_sub_items_self_play(data, kb):
  """process procedure for self play."""
  all_data = tf.string_split([data], sep="|", skip_empty=False).values
  # action is empty for self-play inference
  intent, pred_action, truth_action, utterance, boundary, reward_diag, reward_action = all_data[
      0], all_data[1], all_data[2], all_data[3], all_data[4], all_data[
          5], all_data[6]
  return intent, pred_action, truth_action, kb, utterance, boundary, reward_diag, reward_action


def get_infer_iterator(dataset_data,
                       dataset_kb,
                       vocab_table,
                       batch_size,
                       eod,
                       len_action,
                       output_buffer_size=None,
                       skip_count=None,
                       num_shards=1,
                       shard_index=0,
                       self_play=False):
  """can be used to generate inference or self play iterators."""
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  eod_id = tf.cast(vocab_table.lookup(tf.constant(eod)),
                   tf.int32)  # for padding

  combined_dataset = tf.data.Dataset.zip((dataset_data, dataset_kb))
  combined_dataset = combined_dataset.shard(num_shards, shard_index)

  if skip_count is not None:
    combined_dataset = combined_dataset.skip(skip_count)

  # do not shuffle iterate on inference and self play mode
  # data is shuffled outside of iterator
  combined_dataset = combined_dataset.filter(
      lambda data, kb: tf.logical_and(tf.size(data) > 0,
                                      tf.size(kb) > 0))

  if not self_play:
    get_sub_fu = get_sub_items_infer
    process_entry_fn = partial(process_entry_infer, vocab_table=vocab_table)
  else:
    get_sub_fu = get_sub_items_self_play
    process_entry_fn = partial(process_entry_self_play, vocab_table=vocab_table)

  combined_dataset = combined_dataset.map(get_sub_fu)
  combined_dataset = combined_dataset.map(process_entry_fn)

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  # intent
            tf.TensorShape([]),  # intent_len
            tf.TensorShape([None]),  # source dialogue
            tf.TensorShape([None]),  # target dialogue
            tf.TensorShape([]),  # dialogue_len
            tf.TensorShape([len_action]),  # predicted action
            tf.TensorShape([]),  # action_len
            tf.TensorShape([len_action]),  # trueth  action
            tf.TensorShape([None]),  # reward diag
            tf.TensorShape([len_action]),  # reward action
            tf.TensorShape([None]),  # kb
            tf.TensorShape([]),  # kb_len
            tf.TensorShape([None]),  # mask1
            tf.TensorShape([None]),  # mask2
            tf.TensorShape([None]),  # turn_point
        ),  # action
        padding_values=(
            eod_id,  # src
            0,  # tgt_input
            eod_id,  # source
            eod_id,  # target
            0,
            eod_id,  # predicted action
            0,  # action len
            eod_id,  # truth action
            0.0,  # reward diag
            0.0,  # reward action
            eod_id,  # src_len -- unused
            0,
            False,  # mask 1
            False,  # mask 2
            0.0)  # turn point
    )

  batched_dataset = batching_func(combined_dataset)

  batched_iter = tf.data.make_initializable_iterator(batched_dataset)
  return batched_iter


def get_iterator(dataset_data,
                 dataset_kb,
                 vocab_table,
                 batch_size,
                 t1,
                 t2,
                 eod,
                 len_action,
                 random_seed,
                 num_buckets,
                 max_dialogue_len=None,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0):
  """can be used to generate supervised learning iterators."""
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  eod_id = tf.cast(vocab_table.lookup(tf.constant(eod)), tf.int32)
  t1_id = tf.cast(vocab_table.lookup(tf.constant(t1)), tf.int32)
  t2_id = tf.cast(vocab_table.lookup(tf.constant(t2)), tf.int32)

  combined_dataset = tf.data.Dataset.zip((dataset_data, dataset_kb))
  combined_dataset = combined_dataset.shard(num_shards, shard_index)

  if skip_count is not None:
    combined_dataset = combined_dataset.skip(skip_count)

  combined_dataset = combined_dataset.shuffle(output_buffer_size, random_seed)
  combined_dataset = combined_dataset.filter(
      lambda data, kb: tf.logical_and(tf.size(data) > 0,
                                      tf.size(kb) > 0))

  combined_dataset = combined_dataset.map(get_sub_items_supervised)
  combined_dataset = combined_dataset.map(
      partial(
          process_entry_supervised,
          vocab_table=vocab_table,
          t1_id=t1_id,
          t2_id=t2_id))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  # intent
            tf.TensorShape([]),  # intent_len
            tf.TensorShape([None]),  # source dialogue
            tf.TensorShape([None]),  # target dialogue
            tf.TensorShape([]),  # dialogue_len
            tf.TensorShape([len_action]),  # action
            tf.TensorShape([]),  # action_len
            tf.TensorShape([len_action]),  # pred_action
            tf.TensorShape([None]),  # reward_diag
            tf.TensorShape([len_action]),  # reward_action
            tf.TensorShape([None]),  # kb
            tf.TensorShape([]),  # kb_len
            tf.TensorShape([None]),  # mask1
            tf.TensorShape([None]),  # mask2
            tf.TensorShape([None]),  # turn_point
        ),  # action
        padding_values=(
            eod_id,  # src
            0,  # tgt_input
            eod_id,  # source
            eod_id,  # target
            0,  # diag len
            eod_id,  # action
            0,  # action len
            eod_id,  # pred_action
            0.0,  # reward diag
            0.0,  # reward action
            eod_id,  # kb
            0,  # kb len
            False,  # mask 1
            False,  # mask 2
            0.0)  # turn point
    )

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, unused_4, dialogue_len, unused_6,
                 unused_7, unused_8, unused_9, unused_10, unused_11, unused_12,
                 unused_13, unused_14, unused_15):
      bucket_width = (max_dialogue_len + num_buckets - 1) // num_buckets
      bucket_id = dialogue_len // bucket_width
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = combined_dataset.apply(
        contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(combined_dataset)

  batched_iter = tf.data.make_initializable_iterator(batched_dataset)
  return batched_iter


def get_batched_iterator(iterator):
  """used to generate batching iterators."""
  (intent, intent_len, src_dialogue, tar_dialogue, dialogue_len, action,
   action_len, predicted_action, reward_diag, reward_action, kb,
   has_reservation, mask1, mask2, turns) = (
       iterator.get_next())
  # reshape kb at the end because of padding bs*num_entry, 13
  kb = tf.reshape(kb, [-1, 13])
  return BatchedInput(
      initializer=None,
      intent=intent,
      intent_len=intent_len,
      source=src_dialogue,
      target=tar_dialogue,
      dialogue_len=dialogue_len,
      action=action,
      action_len=action_len,
      predicted_action=predicted_action,  # not used in supervised training
      reward_diag=reward_diag,  # not used in supervised training
      reward_action=reward_action,  # not used in supervised training
      kb=kb,
      has_reservation=has_reservation,
      mask1=mask1,
      mask2=mask2,
      turns=turns)
