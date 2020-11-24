export GLUE_DIR=/home/rthuang/NUT_RC/cls_data
export TASK_NAME=QNLI
export CUDA_VISIBLE_DEVICES=3

python run_train_answer_selection.py \
    --model_type bert \
    --model_name_or_path /home/rthuang/script/pre-bert-model/bert-large-uncased-whole-word-masking \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR \
    --max_seq_length 64 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=12   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps 1000 \
    --output_dir /home/rthuang/debug/cls_large_wwm_debug \
    --overwrite_output_dir
