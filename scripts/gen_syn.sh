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

# default values
mode="airdialogue"
num_samples=400000
resource_path="./data/resources/"
data_path=""
gen_ood1="False"

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -r|--resource) resource_path="$2"
    shift
    shift ;;
    -p|--data_path) data_path="$2"
    shift
    shift ;;
    -n|--num_samples) num_samples="$2"
    shift
    shift ;;
    -s|--synthesized) mode="synthesized"
    shift ;;
    --ood1) gen_ood1="True"
    shift ;;
    *)    # unknown option
    shift ;;
esac
done

# if target path not set, we will assign a path for it
if [ -z $data_path]; then
  data_path="./data/${mode}"
fi
json_path="${data_path}/json"

echo "mode = ${mode}"
echo "num_samples = ${num_samples}"
echo "resource path = ${resource_path}"
echo "data path = ${data_path}"
echo "json path = ${json_path}"

# create directory if not already exist
mkdir -p $json_path

meta1="--firstname_file ${resource_path}/meta_context/first_names.txt \
        --lastname_file ${resource_path}/meta_context/last_names.txt \
        --airportcode_file ${resource_path}/meta_context/airport.txt"

# only generate train and dev data in the synthesized mode
if [ $mode = "synthesized" ]; then
  echo "Generating train data..."
  airdialogue sim \
    ${meta1} \
    --output_data "${json_path}/train_data.json" \
    --output_kb "${json_path}/train_kb.json" \
    --num_samples $((${num_samples}*8/10))

  echo "Generating dev data..."
  airdialogue sim \
    ${meta1} \
    --output_data "${json_path}/dev_data.json" \
    --output_kb "${json_path}/dev_kb.json" \
    --num_samples $((${num_samples}/10))

  echo "Generating test data..."
  airdialogue sim \
    ${meta1} \
    --output_data "${json_path}/test_data.json" \
    --output_kb "${json_path}/test_kb.json" \
    --num_samples $((${num_samples}/10))
fi

# If we process for synthesized or airdialogue we will generate a training
# set for self-play.
# generate train context
echo "Generating selfplay train data..."
airdialogue contextgen \
  ${meta1} \
  --output_data "${json_path}/selfplay_train_data.json" \
  --output_kb "${json_path}/selfplay_train_kb.json" \
  --num_samples $((${num_samples}*8/10))

if [ $gen_ood1 = "True" ]; then
  # If we choose OOD mode we will generate the testing sets for self-play.
  # generating test context for OOD1
  echo "Generating selfplay test data on OOD1..."
  airdialogue contextgen \
    --firstname_file "${resource_path}/meta_context2/first_names.txt" \
    --lastname_file "${resource_path}/meta_context2/last_names.txt" \
    --airportcode_file "${resource_path}/meta_context2/airport.txt" \
    --output_data "${json_path}/ood1_data.json" \
    --output_kb "${json_path}/ood1_kb.json" \
    --num_samples $((${num_samples}/10))
fi

