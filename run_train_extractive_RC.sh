export TWEET_DIR=/home/rthuang/Experimental_result_TweetQA/Norm_SPLT/data
export OUTPUT_DIR=/home/rthuang/Experimental_result_TweetQA/Norm_SPLT/bert-large-wwm-debug
export CUDA_VISIBLE_DEVICES=0

python extractive_RC.py \
  --model_type bert \
  --bert_model /home/rthuang/script/pre-bert-model/bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --ex_train_file $TWEET_DIR/train.json \
  --ex_predict_file $TWEET_DIR/dev.json \
  --overwrite_cache \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --doc_stride 64 \
  --output_dir $OUTPUT_DIR \
  --do_denoising
