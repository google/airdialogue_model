dir=$(pwd)

airdialogue prepro \
    --infer_src_data_file "infer_src_data.json" \
    --infer_kb_file "infer_kb.json" \
    --output_dir "." \
    --output_prefix 'codalab' --job_type '0|0|1|0|0'

cd model
python airdialogue_model_tf.py --task_type INFER --eval_prefix codalab --num_gpus 1 \
                        --infer_src_data "$dir/codalab.infer.src.data" \
                        --infer_kb "$dir/codalab.infer.kb" \
                        --nltk_data "./data/nltk" \
                        --inference_output_file "$dir/inference_out.txt" \
                        --codalab
cd ..
rm codalab.infer.src.data
rm codalab.infer.kb
