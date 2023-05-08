PRE_SEQ_LEN=128
your_data_path=""  # 填入数据集所在的文件夹路径
CHECKPOINT=""   # 填入用来存储模型的文件夹路径
STEP=2000    # 用来评估的模型checkpoint是训练了多少步

CUDA_VISIBLE_DEVICES=2 python medical_prompts/src/ft_chatglm_ptuning/main.py \
    --do_predict \
    --validation_file $your_data_path/dev.json \
    --test_file $your_data_path/dev.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path THUDM/chatglm-6b \
    --ptuning_checkpoint $CHECKPOINT/checkpoint-$STEP \
    --output_dir $CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN
