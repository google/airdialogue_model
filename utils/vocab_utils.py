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

"""Utility to handle vocabularies."""





import codecs
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.lookup import StaticHashTable, TextFileInitializer

UNK = "<unk>"
UNK_ID = 0

start_of_turn1 = "<t1>"
start_of_turn2 = "<t2>"
end_of_dialogue = "<eod>"


def get_vocab_size(vocab_file):
  vocab = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
    for word in f:
      vocab.append(word.strip())
  return len(vocab)


def create_vocab_tables(vocab_file):
  """Creates vocab tables: string > id, then id > string"""
  token_to_id = StaticHashTable(
          TextFileInitializer(vocab_file,
              tf.string,
              tf.lookup.TextFileIndex.WHOLE_LINE,
              tf.int64,
              tf.lookup.TextFileIndex.LINE_NUMBER,
              delimiter=" "), UNK_ID)
  id_to_token = StaticHashTable(
          TextFileInitializer(vocab_file,
              tf.int64,
              tf.lookup.TextFileIndex.LINE_NUMBER,
              tf.string,
              tf.lookup.TextFileIndex.WHOLE_LINE,
              delimiter=" "), UNK)
  return (token_to_id, id_to_token)
