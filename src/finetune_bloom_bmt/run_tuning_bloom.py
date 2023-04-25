import os
import sys

import bmtrain as bmt

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from opendelta import LoraModel, ParallelAdapterModel

sys.path.insert(0, "./")

from further_pretraining.src.finetune_chatglm.arguments import get_args
from medical_prompts.src.finetune_bloom_bmt.tune import CPMAntPlusNLGTune



if __name__ == "__main__":
    args = get_args()

    args.dataset_path = "medical_prompts/datasets/PromptCBLUE/open_version_v0.1"

    data = {}
    data["train"] = os.path.join(args.dataset_path, "train.json")
    data["dev"] = os.path.join(args.dataset_path, "dev.json")

    # load model
    bmt.init_distributed(
        seed=args.seed,
        zero_level=3,
        # loss_scale_factor=2,
        # loss_scale_steps=512,
    )
    config = AutoConfig.from_pretrained(
        "/public/home/xlwang2/codes/Med_Prompts/models--bigscience--bloomz-7b1-mt/snapshots/13e9b1a39fe86c8024fe15667d063aa8a3e32460",
    )
    print(config)
    model = AutoModelForCausalLM.from_pretrained(
        "/public/home/xlwang2/codes/Med_Prompts/models--bigscience--bloomz-7b1-mt/snapshots/13e9b1a39fe86c8024fe15667d063aa8a3e32460",
        config=config,
    ).half()
    print(model)

    model = bmt.BMTrainModelWrapper(model)
    print(model)

    # insert ParallelAdapter
    delta_model = ParallelAdapterModel(
        backbone_model=model,
        modified_modules=['self_attention', "mlp"],
        backend='bmt',
        bottleneck_dim=48,
    )
    delta_model.log()
    delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
    delta_model.log()

    bmt.synchronize()

    # if args.load is not None:
    #     print("load from ", args.load)
    #     bmt.load(model, args.load)

    tokenizer = AutoTokenizer.from_pretrained(
        # "THUDM/chatglm-6b"
        "/public/home/xlwang2/codes/Med_Prompts/models--bigscience--bloomz-7b1-mt/snapshots/13e9b1a39fe86c8024fe15667d063aa8a3e32460"
    )
    print(tokenizer)

    bmt.print_rank("[INFO] Tuning begins...")
    tune = CPMAntPlusNLGTune(
        model=model,
        tokenizer=tokenizer,
        lr=args.lr,
        warmup_iters=args.warmup_iters,
        max_len=args.max_length,
        # cls_num=args.cls_num,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=5,
        eval_interval=args.save_iters,
        output_path=args.save,
    )
    tune.run(data)
    bmt.print_rank("[INFO] Tuning is finished!")

    '''
    
    export CUDA_VISIBLE_DEVICES=2,3
    export CUDA_VISIBLE_DEVICES=0,1
    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12378 medical_prompts/src/finetune_bloom_bmt/run_tuning_bloom.py --lr 1e-5 --max_length 1024 --batch_size 6 --save_iters 500 --save_name finetune_0 --save ./medical_prompts/experiments/instruct_medchat_7b1_mt_1024/ --inspect_iters 20 --warmup_iters 1000 --lr_decay_style noam --weight_decay 1e-3 --clip_grad 2.0 --loss_scale 2048 --start_step 0 --log_dir logs/tensorboard/instruct_medchat_7b1_mt_1024/ --load /public/home/xlwang2/codes/Med_Prompts/checkpoints/bianque_bloomz_7b1_mt_1024/bianque_checkpoint-100000.pt --epochs 5
    
    
    '''
