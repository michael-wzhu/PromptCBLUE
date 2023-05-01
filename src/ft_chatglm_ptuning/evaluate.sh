PRE_SEQ_LEN=128
CHECKPOINT=./medical_prompts/experiments/output/adgen-chatglm-6b-pt-128-2e-2
STEP=2000

CUDA_VISIBLE_DEVICES=1 python medical_prompts/src/ft_chatglm_ptuning/main.py \
    --do_predict \
    --validation_file medical_prompts/datasets/PromptCBLUE/internal/gt/dev.json \
    --test_file medical_prompts/datasets/PromptCBLUE/internal/gt/dev.json \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path ./models--THUDM--chatglm-6b/snapshots/a8ede826cf1b62bd3c78bdfb3625c7c5d2048fbd \
    --ptuning_checkpoint $CHECKPOINT/checkpoint-$STEP \
    --output_dir $CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 700 \
    --max_target_length 196 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN
