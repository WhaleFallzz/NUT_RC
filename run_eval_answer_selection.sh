# 数据目录
export GLUE_DIR=/home/rthuang/Experimental_result_TweetQA/cls_data/generation_ans
# export GLUE_DIR=/home/rthuang/NUT_RC/data/cls_data
# 输出目录
export ANS_TYPE=Gen_RC_dev
# 引用的QNLI的代码，不用管
export TASK_NAME=QNLI
export CUDA_VISIBLE_DEVICES=3

python run_eval_answer_selection.py \
    --model_type bert \
    --model_name_or_path /home/rthuang/debug/cls_large_wwm \
    --task_name $TASK_NAME \
    --ans_type $ANS_TYPE \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR \
    --input_file $GLUE_DIR/dev.tsv \
    --max_seq_length 64 \
    --per_gpu_eval_batch_size=48 \
    --output_dir /home/rthuang/debug/cls_large_wwm \
    --overwrite_output_dir
