export DATA_DIR=/home/rthuang/Experimental_result_TweetQA/Norm_SPLT/gqa_data/dev
export MODEL_RECOVER_PATH=/home/rthuang/Experimental_result_TweetQA/Norm_SPLT/multi_task/pytroch_model.bin
# export OUTPUT_DIR=/home/rthuang/Experimental_result_TweetQA/Norm_SPLT/multi_task
export OUTPUT_DIR=/home/rthuang/NUT_RC/cls_data
export EVAL_FILE=/home/rthuang/TweetQA/TweetQA_data/dev.json
export CUDA_VISIBLE_DEVICES=2

python run_unilm_decode.py \
  --bert_model /home/rthuang/script/pre-bert-model/bert-large-uncased-whole-word-masking \
  --do_lower_case \
  --new_segment_ids --mode s2s \
  --input_file $DATA_DIR/dev.pq.tok.txt \
  --qid_file $DATA_DIR/dev.qid.txt \
  --eval_file $EVAL_FILE \
  --tokenized_input \
  --model_recover_path $MODEL_RECOVER_PATH \
  --max_seq_length 128 --max_tgt_length 24 \
  --batch_size 48 --beam_size 1 --length_penalty 0 \
  --output_path $OUTPUT_DIR \
  --split gen_RC_dev
