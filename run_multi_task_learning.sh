##########################################################################
# File Name: run_multi_task.sh
# Author: amoscykl
# mail: amoscykl980629@163.com
# Created Time: Thu 14 Nov 2019 06:29:20 PM CST
#########################################################################
#!/bin/zsh
# PATH=/home/edison/bin:/home/edison/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/work/tools/gcc-3.4.5-glibc-2.3.6/bin
# export PATH

export DATA_DIR=/home/rthuang/Experimental_result_TweetQA/Norm_EXPAN_MISC/gqa_data/train
export OUTPUT_DIR=/home/rthuang/Experimental_result_TweetQA/Norm_EXPAN_MISC/multi_task_debug
export MODEL_RECOVER_PATH=/home/rthuang/script/pre-bert-model/bert-large-uncased-whole-word-masking/pytorch_model.bin
export CUDA_VISIBLE_DEVICES=2

python multi_task_learning.py \
  --do_train \
  --bert_model /home/rthuang/script/pre-bert-model/bert-large-uncased-whole-word-masking \
  --do_lower_case \
  --new_segment_ids \
  --tokenized_input \
  --data_dir $DATA_DIR --src_file train.pq.tok.txt --tgt_file train.a.tok.txt \
  --output_dir $OUTPUT_DIR \
  --log_dir $OUTPUT_DIR \
  --model_recover_path $MODEL_RECOVER_PATH \
  --max_seq_length 128 --max_position_embeddings 128 \
  --mask_prob 0.7 --max_pred 24 \
  --train_batch_size 12 \
  --gradient_accumulation_steps 2 \
  --learning_rate 0.00002 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 10
