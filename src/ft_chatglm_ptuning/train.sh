PRE_SEQ_LEN=128
LR=2e-2
your_data_path=""  # 填入数据集所在的文件夹路径
your_checkpopint_path=""  # 填入用来存储模型的路径

ptuning_checkpoint=""  # 如果之前训练过，且存储了ptuning权重，则设置为ptuning权重的文件夹路径

CUDA_VISIBLE_DEVICES=1 python src/ft_chatglm_ptuning/main.py \
    --do_train \
    --train_file $your_path/train.json \
    --validation_file $your_path/dev.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir $your_checkpopint_path/PromptCBLUE-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --max_steps 10000 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --ptuning_checkpoint $ptuning_checkpoint



