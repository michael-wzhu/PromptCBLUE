PRE_SEQ_LEN=128
LR=2e-2
your_data_path="./datasets/PromptCBLUE/toy_examples/"  # 填入数据集所在的文件夹路径
your_checkpopint_path="./experiments/outputs/"  # 填入用来存储模型的路径
model_name_or_path="./models--THUDM--chatglm-6b/snapshots/a8ede826cf1b62bd3c78bdfb3625c7c5d2048fbd"   # LLM底座模型路径，或者是huggingface hub上的模型名称


CUDA_VISIBLE_DEVICES=1 python src/ft_chatglm_ptuning/main.py \
    --do_train \
    --train_file $your_data_path/train.json \
    --validation_file $your_data_path/dev.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpopint_path/PromptCBLUE-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_steps 10000 \
    --logging_steps 10 \
    --save_steps 10 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN



