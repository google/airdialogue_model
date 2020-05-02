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

#!/bin/bash
set -e # stop script when there is an error

# mode can be either synthesized, airdialogue or OOD.

IFS=':' # space is set as delimiter

mode="airdialogue"
word_cutoff=10
nltk_data="./data/nltk"
data_path=""
additional_str=""

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -n|--nltk_path) nltk_path="$2"
    shift
    shift ;;
    -d|--data_path) data_path="$2"
    shift
    shift ;;
    -p|--prefix) partition="$2"
    shift
    shift ;;
    -w|--word_cutoff) word_cutoff="$2"
    shift
    shift ;;
    -a) additional_str=${2}
    shift
    shift ;;
    -s|--synthesized) mode="synthesized"
    shift ;;
    *)    # unknown option
    shift ;;
esac
done

# if target path not set, we will assign a path for it
if [ -z $data_path]; then
  data_path="./data/${mode}"
fi

echo "mode = ${mode}"
echo "word cutoff = ${word_cutoff}"
echo "nltk data path = ${nltk_data}"
echo "data path = ${data_path}"
echo "partition = ${partition}"

json_path="${data_path}/json"
tokenized_path="${data_path}/tokenized"
# create directory if not already exist
mkdir -p $tokenized_path
echo "json path = ${json_path}"
echo "tokenized path = ${tokenized_path}"
read -ra ADDS <<< "$additional_str"


# We process train, dev and test sets for supervised learning.
# Note that we will also generate a copy of the dev and test set for infer and
# self-play evaluations.

if [[ $partition = "train" ]]; then
  job_type_str="1|0|0|0|0"
  cutoff_flag="10"
  gen_voc_flag="--gen_voc"
else
  job_type_str="0|1|1|0|1"
  cutoff_flag="0"
  gen_voc_flag=""
fi

echo "tokenizing ${partition} data..."
airdialogue prepro \
  --data_file "${json_path}/${partition}_data.json" \
  --kb_file "${json_path}/${partition}_kb.json" \
  --output_dir "${tokenized_path}" \
  --output_prefix ${partition} --job_type ${job_type_str} --input_type dialogue \
  --nltk_data ${nltk_data} --word_cutoff ${cutoff_flag} \
  $gen_voc_flag

if [[ $partition = "train" ]]; then
  # tokenizing context
  echo "tokenizing selfplay train data"
  airdialogue prepro \
    --data_file "${json_path}/selfplay_train_data.json" \
    --kb_file "${json_path}/selfplay_train_kb.json" \
    --output_dir "${tokenized_path}" \
    --output_prefix 'train' --job_type '0|0|0|1|0' --input_type context
fi

# If we choose OOD mode we will generate the testing sets for self-play.
# generating test context for OOD

# tokenizing context for OOD1
for i in "${ADDS[@]}"; do
  echo "tokenizing self-play evaluation on $i"
  airdialogue prepro \
    --data_file "${json_path}/${i}_data.json" \
    --kb_file "${json_path}/${i}_kb.json" \
    --output_dir "${tokenized_path}" \
    --output_prefix "${i}" --job_type '0|0|0|0|1' --input_type context
done
