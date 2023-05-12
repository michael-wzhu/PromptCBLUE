

## 依赖

我们根据我们实验所用的环境生成了[requirements.txt](./requirements.txt)。参赛者可以自行配置版本更高的环境。

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

```bash
./src/ft_chatglm_lora/evaluate.sh

```