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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import tensorflow as tf

from tensorflow.python.ops import lookup_ops

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
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  return lookup_ops.index_table_from_file(vocab_file, default_value=UNK_ID)
