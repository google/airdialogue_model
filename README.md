
# AirDialogue Model
- Official implementations of the [AirDialogue paper][paper]
- Codebase developed based on the [AirDialogue tookit][airdialogue]
- Replicating results by using the [AirDialogue dataset][data], or the [synthesized dataset](#markdown-header-working-with-synthesized-dataset)

## Prerequisites
#### General
- python (verified on 3.7)
- wget

#### Python Packages
- tensorflow (verified on 1.15.0)
- airdialogue

## 1. Prepare Dataset
AirDialogue dataset and its meta data can be downloaded using our download script. In this script we will also download the nltk corpus used for preprocessing.
```
bash ./scripts/download.sh
```
We will also generate a set of synthesized context pairs for self-play training. These context pairs contain initial conditions and the optimal decisions of the synthesized dialogue. Additionally, here we also generate the Out-of-domain evaluation set (OOD1). See the [AirDialogue paper][paper] for more details.

```
bash ./scripts/gen_syn.sh --ood1
```
## 2. Preprocessing
We preprocess the train dataset in order to begin the training of our model.
```
bash ./scripts/preprocess.sh -p train
```

## 3. Training
#### Supervised Learning
The fist step is to train our model using supervised learning.
```
python airdialogue_model_tf.py --task_type TRAINEVAL --num_gpus 8 \
                            --input_dir ./data/airdialogue/tokenized \
                            --out_dir ./data/out_dir \
                            --num_units 256 --num_layers 2
```
#### Training with Reinforcement Learning Self-play
The second step would be to train our model using self-play based on our supervised learning checkpoint.
```
python airdialogue_model_tf.py --task_type SP_DISTRIBUTED --num_gpus 8 \
                            --input_dir ./data/airdialogue/tokenized \
                            --self_play_pretrain_dir ./data/out_dir \
                            --out_dir ./data/selfplay_out_dir
```
#### Examine Training Meta Information
Training meta data will be written to the output directory, which can be examined using `tensorboard`. The following command will examine the training procedure of the supervised learning model.
```
tensorboard --logdir=./data/out_dir
```
To view the training meta data for the self-play mode, swap logdir to `./data/selfplay_out_dir`.
## 4. Evaluating on the AirDialogue dev set
#### Preprocessing
Similar to training, we will first need to preprocess the dev dataset. here we will also preprocess the ood1 dataset for evaluation.
```
bash ./scripts/preprocess.sh -p dev --ood1
```
#### Predicting
We use the following script to evaluate our trained model on the dev set. Following the [AirDialogue paper][paper], here we also evaluate the model's performance on the OOD1 evaluation set that we generated. The evaluation script will first try to find the selfplay model. If failed, it will use the supervised model.
```
bash ./scripts/evaluate.sh -p dev -a ood1
```
## 5. Scoring
Once the predictative files are generated, we will depend on the [AirDialogue tookit][airdialogue] for scoring.
We are currently working on the scoring script.
```
airdialogue score --pred_data ./data/out_dir/dev_inference_out.txt \
                  --true_data ./data/airdialogue/tokenized/dev.infer.tar.data \
                  --true_kb ./data/airdialogue/tokenied/dev.infer.kb \
                  --task infer \
                  --output ./data/out_dir/dev_bleu.json
```
```
airdialogue score --pred_data ./data/out_dir/dev_selfplay_out.txt \
                  --true_data ./data/airdialogue/json/dev_data.json \
                  --true_kb ./data/airdialogue/json/dev_kb.json \
                  --task selfplay \
                  --output ./data/out_dir/dev_selfplay.json
```
```
airdialogue score --pred_data ./data/out_dir/ood1_selfplay_out.txt \
                  --true_data ./data/airdialogue/json/ood1_data.json \
                  --true_kb ./data/airdialogue/json/ood1_kb.json \
                  --task selfplay \
                  --output ./data/out_dir/ood1_selfplay.json
```

## 6. Evaluating on the AirDialogue test set
We are currently working on the evalaution process of the test set.

## 7. Benchmark Results
We are currently working on benchmarking the results.

## 8. Misc
#### a. Task and Dataset Alignments

|Stage|||Tasks|||
|----|----|----|---|---|---|
|**Training**|**Supervised**|**Self-play**|
||train.data|train.selfplay.data|
|          |train.kb|train.selfplay.kb|
|source    |Airdialogue|synthesized (meta1)|
|**Testing-Dev**|**Inference**             |**Self-play Eval**            |**Self-play Eval**    |**Eval**         |
||dev.infer.src.data|dev.selfplay.eval.data|ood1.selfplay.data|dev.eval.data|
|           |dev.infer.tar.data|                      |         |             |
|           |dev.infer.kb      |dev.selfplay.eval.kb  |ood1.selfplay.eval.kb  |dev.eval.kb  |
|source    |AirDialogue  |AirDialogue       |synthesized (meta2)|AirDialogue|
|**Testing-Test (hidden)**|**Inference**             |**Self-play Eval**            |**Self-play Eval**    |
| |test.infer.src.data|test.selfplay.eval.data|ood2.selfplay.data|
|             |test.infer.tar.data|                      |         |
|             |test.infer.kb      |test.selfplay.eval.kb  |ood2.selfplay.kb  |
|source    |AirDialogue  |AirDialogue       |synthesized (meta3)|


#### b. Working with Synthesized Data
As an alternative to the AirDialogue Dataset, we can verify our model using the synthesized data.
###### Training
To genreate a synthesized dataset for training, flip the -s option for the data generation script. By default, synthesized data will be put under `./data/synthesized/`
```
bash ./scripts/gen_syn.sh -s --ood1
```
We will then need to preprocess the synthesized data for training
```
bash ./scripts/preprocess.sh -s -p train
```

Similar to experiments on the AirDialogue dataset, we can train a supervised model for the synthesized data:
```
python airdialogue_model_tf.py --task_type TRAINEVAL --num_gpus 8 \
                            --input_dir ./data/synthesized/tokenized \
                            --out_dir ./data/synthesized_out_dir \
                            --num_units 256 --num_layers 2

```
With supervised model pre-training, we can also train the synthesized model using self-play:
```
python airdialogue_model_tf.py --task_type SP_DISTRIBUTED --num_gpus 8 \
                            --input_dir ./data/synthesized/tokenized \
                            --self_play_pretrain_dir ./data/synthesized_out_dir \
                            --out_dir ./data/synthesized_selfplay_out_dir
```
###### Testing
Before testing on the dev data, we will need to do preprocessing.
Dev Dataset
```
bash ./scripts/preprocess.sh -p dev --ood1 -s
```
We can run execute the evalution script on the synthesized dev set.
```
bash ./scripts/evaluate.sh -p dev -a ood1 -m ./data/synthesized_out_dir -o ./data/synthesized_out_dir -i ./data/synthesized/tokenized/
```
###### Scoring
```
airdialogue score --pred_data ./data/synthesized_out_dir/dev_inference_out.txt \
                  --true_data ./data/synthesized/tokenized/dev.infer.tar.data \
                  --true_kb ./data/airdialogue/tokenized/dev.infer.kb \
                  --task infer \
                  --output ./data/synthesized_out_dir/dev_bleu.json
```
```
airdialogue score --pred_data ./data/synthesized_out_dir/dev_selfplay_out.txt \
                  --true_data ./data/synthesized/json/dev_data.json \
                  --true_kb ./data/airdialogue/json/dev_kb.json \
                  --task selfplay \
                  --output ./data/synthesized_out_dir/dev_selfplay.json
```
```
airdialogue score --pred_data ./data/synthesized_out_dir/ood1_selfplay_out.txt \
                  --true_data ./data/synthesized/json/ood1_data.json \
                  --true_kb ./data/airdialogue/json/ood1_kb.json \
                  --task selfplay \
                  --output ./data/synthesized_out_dir/ood1_selfplay.json
```
One can repeat same steps for synthesized test set as well. Please refer to the [AirDialogue paper][paper] for the results on the synthesized dataset.

[data]: https://storage.googleapis.com/airdialogue/airdialogue_data.tar.gz
[paper]: https://www.aclweb.org/anthology/D18-1419/
[airdialogue]: https://github.com/google/airdialogue
