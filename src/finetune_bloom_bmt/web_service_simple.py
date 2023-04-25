import json
import time

import bmtrain as bmt
import torch
from opendelta import AutoDeltaModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

bmt.init_distributed(
    seed=0,
    zero_level=3,   # support 2 and 3 now
    # ...
)

# model_path = "/public/home/xlwang2/codes/Med_Prompts/models--BelleGroup--BELLE-7B-2M/snapshots/a9076d928eff1d94fe6b4372ba2bd3a800dc10a1"
model_path = "/public/home/xlwang2/codes/Med_Prompts/models--bigscience--bloomz-7b1-mt/snapshots/13e9b1a39fe86c8024fe15667d063aa8a3e32460"

config = AutoConfig.from_pretrained(
        model_path,
    )
print(config)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
).half()
print(model)

model = bmt.BMTrainModelWrapper(model)
print(model)

# load_path = "./medical_prompts/experiments/instruct_medchat_7b1_mt_1024/best.pt"
# bmt.load(model, load_path)

delta_model = AutoDeltaModel.from_finetuned(
    "./medical_prompts/experiments/instruct_medchat_7b1_mt_1024/best.pt",
    backbone_model=model
)

bmt.synchronize()

# import bminf
# with torch.cuda.device(0):
#     model = bminf.wrapper(
#         model,
#         quantization=False
#     )

#

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
)

# # # inference
# input_ids = tokenizer(
#     "问：男，目前28岁，最近几年，察觉，房事不太给力，另外，每次才开始就已经射了，请问：男生早泄是由于哪些原因诱发的。\n答：",
#     # "问：颈柱病是如何引起的\n我是坐办公室的，经常有不明因素或病因的落枕，最近还有耳鸣，不太严重。这是不是颈柱问题，怎么预防。\n答：",
#     # "问：给出句子中的实体：\n右心室如有衰竭而扩张，在心前区有广泛的搏动，甚至可延伸到腋前线。\n医学实体选项: 疾病, 身体部位, 临床表现\n答：",
#     return_tensors="pt",
#     add_special_tokens=False,
#     # pad_to_multiple_of=8,
#
# ).input_ids  # Batch size 1
# print(input_ids)
#
# t0 = time.time()
# with torch.no_grad():
#     # streamer = TextStreamer(tokenizer)
#     outputs = model.generate(input_ids.cuda(), max_new_tokens=256)
#
# print(outputs)
#
# t1 = time.time()
# print("time cost: ", t1 - t0)
#
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


from flask import Flask, request

app = Flask(__name__)


@app.route("/instruct_generate", methods=["POST"])
def cough_predict():
    input_data = json.loads(
        request.get_data().decode("utf-8")
    )

    query = input_data.get("query")
    max_new_tokens = input_data.get("max_new_tokens", 256)

    input_ids = tokenizer(
        query,
        return_tensors="pt",
        add_special_tokens=False,
        # pad_to_multiple_of=8,

    ).input_ids  # Batch size 1

    t0 = time.time()
    with torch.no_grad():
        # streamer = TextStreamer(tokenizer)
        outputs = model.generate(
            input_ids.cuda(),
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2
        )

    print(outputs)

    t1 = time.time()
    print("time cost: ", t1 - t0)

    generated_sents = []
    for out in outputs:
        generated_sents.append(tokenizer.decode(out, skip_special_tokens=True))

    return {
        "sentence": generated_sents
    }


app.run(host="0.0.0.0", port=9001, debug=False)


'''

export CUDA_VISIBLE_DEVICES=2
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12346 medical_prompts/src/finetune_bloom_bmt/web_service_simple.py




'''
