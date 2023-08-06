

## 依赖

我们根据我们实验所用的环境生成了[requirements.txt](./requirements.txt)。参赛者可以自行配置版本更高的环境。

## 模型下载

先将ChatGLM权重下载到本地。通过下面的命令，
```bash
python src/download_checkpoints.py
```
模型会存放在类似于`./models--THUDM--chatglm-6b/snapshots/a8ede826cf1b62bd3c78bdfb3625c7c5d2048fbd`的路径中，加载模型时候就是采用这个路径。


##  数据

请前往[PromptCBLUE通用赛道评测网站](https://tianchi.aliyun.com/competition/entrance/532085/introduction)或者[PromptCBLUE通用赛道评测网站](https://tianchi.aliyun.com/competition/entrance/532084/introduction)下载训练集，验证集以及测试集A或者测试集B。这些数据放置在自己指定的文件夹中，如"datasets/PromptCBLUE/toy_examples"。


## ChatGLM-6B + P-tuning 方法

这部分代码借鉴了ChatGLM-6B官方的p-tuning代码。

### 训练

```bash
./src/ft_chatglm_ptuning/train.sh

```


### 预测(生成回复)

```bash
./src/ft_chatglm_ptuning/evaluate.sh

```



## ChatGLM-6B + LoRA方法微调

这部分代码实现借助了[PEFT项目](https://github.com/huggingface/peft)。注意PEFT直接用pip安装的话，需要torch==2.0以上，同时cuda也需要高版本。如果大家不想更新torch环境，可以直接拷贝他的核心代码，放在自己的代码库里面，如[./src/ft_chatglm_lora/peft](./src/ft_chatglm_lora/peft)，这样就可以在更低版本的torch环境下使用。

注意ChatGLM-6B采用了query，key，value矩阵参数共享，所以LoRA作用的模块名称是与其他模型不同的。我们这里要LoRA作用于`query_key_value,dense,dense_h_to_4h,dense_4h_to_h`这些模块。


### 训练

```bash

src/ft_chatglm_lora/train.sh

```

### 预测(生成回复)

预测时，可以根据自身判断，选择调整`src/ft_chatglm_lora/main.py`代码的445行到455行的模型生成设置，比如`num_beams`, `do_sample`等。我们现在设置`do_sample=False`和`num_beams=1`，即采用贪心解码。自然地，设置更大的`num_beams`相应的可以提升生成效果，不过也会带来显存压力。

同时，大家根据卡的显存，设置下面脚本中的`per_device_eval_batch_size`取值。我们目前的生成设置和脚本设置的入参，推理需要25G显存，在V100 (40G)上单卡5个小时左右跑完测试集。

```bash
./src/ft_chatglm_lora/evaluate.sh

```

预测效率提升有很多途径：包括模型量化，或者使用推理框架，如vLLM。



## LlaMA-7B + LoRA方法微调

### LlaMA模型准备

我们先要准备LlaMA模型底座，使得其可以在huggingface transformers框架下进行参数高效微调。准备工作主要有三步：

#### LlaMA模型主干

获取LlaMA模型主干有几种途径：
- 原版LLaMA模型: 在[LlaMA原项目地址](https://github.com/facebookresearch/llama)填写google form申请;
- [LlaMA项目的一个PR](https://github.com/facebookresearch/llama/pull/73/files)
- huggingface的model hub中已经人上传了模型: [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)

#### LlaMA模型权重转化

上一步骤的前两种方法需要将LlaMA模型权重转化为huggingface transformers的格式，详见[convert_llama_weights_to_hf](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py))。


#### 融合Chinese-LlaMA-Alpaca

[Chinese-LlaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md)项目提供了使得LlaMA模型更适应于中文场景的lora权重和经过继续预训练的embedding权重。我们采用其脚本将其权重合并到模型主干中：

```bash
python src/ft_llama_lora/merge_llama_with_chinese_lora.py \
    --base_model decapoda-research/llama-7b-hf \
    --lora_model ziqingyang/chinese-llama-plus-lora-7b,ziqingyang/chinese-alpaca-plus-lora-7b \
    --output_type huggingface \
    --output_dir ./resources/chinese-llama-alpaca-plus-lora-7b

```

注意上述命令中我们合并了[Chinese-LlaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)的两个lora权重，第一个权重是做了大规模中文语料预训练，第二个权重则是进一步做了基于self-instruct的中文指令微调。两者合并可以得到更会说中国话的LlaMA模型。

注意存储合并参数后的llama模型时，尽量模型文件shards多分一些,比如`max_shard_size=2GB`, 这样加载起来也会快一些。


### 训练

```bash

CUDA_VISIBLE_DEVICES="2,3" ./my_scripts/promptcblue_fft/run_train.sh

```

### 预测(生成回复)

预测时，我们采用[vllm项目](https://github.com/vllm-project/vllm)对模型进行serving.同时这部分代码参照了[KnowLM项目](https://github.com/zjunlp/KnowLM/tree/main/inference)

在使用vllm时，我们首先要把训练得到的lora参数与LlaMA主干进行合并 (假设我们采用训练第800步的lora权重)：

```bash

CUDA_VISIBLE_DEVICES="3" python src/ft_llama_lora/merge_llama_with_chinese_lora.py \
    --base_model ./resources/chinese-llama-plus-lora-7b \
    --lora_model ./experiments/output/promptcblue-llama-7b-pt-v0/checkpoint-800 \
    --output_type huggingface \
    --output_dir ./experiments/output/promptcblue-llama-7b-pt-v0/checkpoint-800-merge

```

然后采用下面的命令启动模型服务。注意，我们修改了`src/ft_llama_lora/vllm_serving/llm_engine.py`第148行的`gpu_memory_utilization`参数取值，大家可以根据显卡情况修改。

```bash
CUDA_VISIBLE_DEVICES="3" python src/ft_llama_lora/vllm_serving/launch_vllm.py \
    --port 8000 \
    --model ./experiments/output/promptcblue-llama-7b-pt-v0/checkpoint-800-merge \
    --use-np-weights \
    --max-num-batched-tokens 4096 \
    --dtype half \
    --tensor-parallel-size 1

```

我们在生成的时候，不会传入有效的`parameters`字段，所以采样参数会使用`src/ft_llama_lora/vllm_serving/launch_vllm.py`的63行处`SamplingParams`的默认值。大家可以根据需求修改。vllm服务起好之后，我们可以通过下面的例子进行服务调用，从而进行测试集预测：

```bash
python src/ft_llama_lora/vllm_serving/web_service_test.py

```

通过vllm部署模型，我们测试下来预计加速2.5倍左右。




## Contributors

- [michael-wzhu](https://github.com/michael-wzhu)
- [boom-R123](https://github.com/boom-R123)
