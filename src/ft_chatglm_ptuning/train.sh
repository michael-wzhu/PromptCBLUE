PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=1 python medical_prompts/src/ft_chatglm_ptuning/main.py \
    --do_train \
    --train_file medical_prompts/datasets/PromptCBLUE/open_version_v0.1/train.json \
    --validation_file medical_prompts/datasets/PromptCBLUE/open_version_v0.1/dev.json \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path ./models--THUDM--chatglm-6b/snapshots/a8ede826cf1b62bd3c78bdfb3625c7c5d2048fbd \
    --output_dir ./medical_prompts/experiments/output/PromptCBLUE-chatglm-6b-pt-$PRE_SEQ_LEN-$LR \
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
    --ptuning_checkpoint ./medical_prompts/experiments/output/chatmed-chatglm-6b-pt-128-2e-2-v0/checkpoint-3000



# --quantization_bit 4
# predict_with_generate

# /public/home/xlwang2/install_files/cuda-11.8