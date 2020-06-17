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

# process commandline flags
# dataset=()
# airdialogue_evaluation=('INFER' 'SP_EVAL')
IFS=':' # space is set as delimiter
partition=""
additional_str=""

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -i|--input_dir) input_dir="$2"
    shift
    shift ;;
    -o|--out_dir) out_dir="$2"
    shift
    shift ;;
    -m|--model_dir) model_dir="$2"
    shift
    shift ;;
    -g|--gpu) num_gpus=$2 # set to 0 to disable GPU
    shift
    shift ;;
    -p|--prefix) partition="$2"
    shift
    shift ;;
    -a) additional_str=${2}
    shift
    shift ;;
    *)    # unknown option
    shift ;;
esac
done

num_gpus=${num_gpus:-1} # by default use one gpu
background=${background:-"False"}
input_dir=${input_dir:-"./data/airdialogue/tokenized"}
if [ -d "./data/selfplay_out_dir" ]; then
  default_dir="./data/selfplay_out_dir"
else
  default_dir="./data/out_dir"
fi
out_dir=${out_dir:-${default_dir}}
read -ra ADDR <<< "$additional_str"
echo "out_dir", ${out_dir}
echo "num_gpus", ${num_gpus}

# run in foreground once and display the results
python airdialogue_model_tf.py --task_type INFER --eval_prefix $partition --num_gpus $num_gpus \
                      --input_dir ${input_dir} --out_dir ${out_dir} \
                      --inference_output_file ${out_dir}/dev_inference_out.txt

# run in foreground once and display the results
python airdialogue_model_tf.py --task_type SP_EVAL --eval_prefix $partition --num_gpus $num_gpus \
                        --input_dir ${input_dir} --self_play_pretrain_dir ${out_dir} \
                        --self_play_immutable_gpu --self_play_eval_batch_size 256 \
                        --selfplay_eval_output_file ${out_dir}/dev_selfplay_out.txt \
                        --out_dir ${out_dir}

# additional evaluation
for task in ${ADDR[@]}
do
  # run in foreground once and display the results
  python airdialogue_model_tf.py --task_type SP_EVAL --eval_prefix ${task} --num_gpus $num_gpus \
                          --input_dir ${input_dir} --self_play_pretrain_dir ${out_dir} \
                          --self_play_immutable_gpu --self_play_eval_batch_size 256 \
                          --selfplay_eval_output_file ${out_dir}/${task}_selfplay_out.txt \
                          --out_dir ${out_dir}

done
