lora_rank=8
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="THUDM/chatglm-6b"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="./datasets/PromptCBLUE/toy_examples"  # 填入数据集所在的文件夹路径
CHECKPOINT="./experiments/outputs/PromptCBLUE-chatglm-6b-lora-2e-4"   # 填入用来存储模型的文件夹路径
STEP=3000    # 用来评估的模型checkpoint是训练了多少步

CUDA_VISIBLE_DEVICES=2 python src/ft_chatglm_lora/main.py \
    --do_predict \
    --validation_file $your_data_path/dev.json \
    --test_file $your_data_path/dev.json \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path $CHECKPOINT/checkpoint-$STEP \
    --output_dir $CHECKPOINT/checkpoint-$STEP \
    --overwrite_output_dir \
    --max_source_length 828 \
    --max_target_length 196 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate
