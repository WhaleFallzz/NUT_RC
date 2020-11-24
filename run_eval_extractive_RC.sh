  # --do_save \
  # --save_step 500 \

# 测试数据
export TWEET_DIR=/home/rthuang/Experimental_result_TweetQA/Norm_SPLT/data
# 测试结果
export OUTPUT_DIR=/home/rthuang/Experimental_result_TweetQA/Norm_SPLT/bert-large-wwm
export CUDA_VISIBLE_DEVICES=2
# "--bert_model"训练好的模型（run_train_extract.sh）的目录
python extractive_RC.py \
  --model_type bert \
  --bert_model /home/rthuang/Experimental_result_TweetQA/Norm_SPLT/bert-large-wwm-finetuned \
  --do_eval \
  --do_lower_case \
  --ex_train_file $TWEET_DIR/train.json \
  --ex_predict_file $TWEET_DIR/train.json \
  --overwrite_cache \
  --per_gpu_train_batch_size 48 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 128 \
  --doc_stride 64 \
  --output_dir $OUTPUT_DIR