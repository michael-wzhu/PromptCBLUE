[**中文**](./README.md) | [**English**](./README_EN.md) 

<p align="center">
    <br>
    <img src="./pics/promptCBLUE_banner_v0.png" width="600"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca.svg?color=blue&style=flat-square">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca">
</p>


以ChatGPT、GPT-4等为代表的大语言模型（Large Language Model, LLM）掀起了新一轮自然语言处理领域的研究浪潮，展现出了类通用人工智能（AGI）的能力，受到业界广泛关注。在LLM大行其道的背景下，几乎所有的NLP任务都转化为了基于提示的语言生成任务。然而，在中文医学NLP社区中，尚未有一个统一任务形式的评测基准。

为推动LLM在医疗领域的发展和落地，华东师范大学王晓玲教授团队联合天池推出**PromptCBLUE**评测基准:

- 🚀 将[CBLUE](https://tianchi.aliyun.com/dataset/95414)基准进行二次开发，将16种不同的医疗场景NLP任务全部转化为基于提示的语言生成任务,形成首个中文医疗场景的LLM评测基准，有利于帮助开源社区和业界快速评测公开的或者LLM模型

同时，为辅助LLM在医疗领域的各项能力提升，我们同时开源以下数据/模型资源供参赛者使用：
- 🚀 **PromptCBLUE**的baseline模型[ChatMed-instruct](xxx)，模型主干来自开源的Bloom-7b1-mt模型，微调时采用全量微调或者是Parallel-Adapter参数高效微调，在PromptCBLUE训练集上进行微调。代码见[code](./src/finetune_bloom_bmt)。
- 🚀 为提升LLM的医疗知识与回答医学咨询的能力，开源[中文医疗问答数据集ChatMed-Datasets](https://huggingface.co/datasets/michaelwzhu/ChatMed-Datasets)。本数据集采用在线爬取的50w在线问诊数据作为prompt，调用ChatGPT得到答案。本数据集真实反映了真实世界的在线问诊需求。


----

[Text2DT](https://github.com/michael-wzhu/Text2DT_Baseline) | [中文医疗大模型ChatMed](https://github.com/michael-wzhu/ChatMed) | [中文医疗问答数据集ChatMed-Datasets](https://huggingface.co/datasets/michaelwzhu/ChatMed-Datasets)


## 更新

2023/4/25 PromptCBLUE(v0.1)上线了，将持续更新！ 🎉🎉🎉




## 数据集详情

### PromptCBLUE总体统计



| PromptCBLUE      | -     |
|-------------|-------|
| 版本号         | v0.1  |
| prompt 模板数量 | 94    |
| 训练集         | 68500 |
| 验证集         | 10270 |
| 测试集A        | 10270 |
| 测试集B        | ⏳     |




### CBLUE任务改造

我们采用xxx个指令微调模板，对CBLUE基准中的各个任务进行。经过改造后，医疗文本NLP数据集都将转化为如下格式。input字段是模型的输入，target字段是模型的输出，type是原任务类型(不作为模型输入)，answer_choices字段是选项，只有分类、术语标准化、推理类任务上该字段才会有意义。

```bash
{
  "input": str,
  "target": str,
  "type": str,
  "answer_choices": str,
  "sample_id": str,
}
```

为了将CBLUE中的各种不同任务适配为符合LLM的输入输出格式，我们对CBLUE各个数据集进行了相应的改造。详见[CBLUE任务改造](./src/data/CBLUE任务改造说明与举例.md)



## 评测方案

### 评测平台

- PromptCBLUE将作为CCKS-2023的评测任务之一，依托天池大赛平台举办第一期评测；
- 在CCKS-2023会议后，PromptCBLUE将作为天池数据集，长期开放榜单，供社区评测/打榜。

### 提交格式说明

**注意** 本项目中的datasets/toy_examples文件夹只为帮助说明文件格式。参赛选手需要到[PromptCBLUE天池页面](xxx)进行完整数据集下载。

PromptCBLUE中将各个任务都已经转化为了根据prompt生成回复的统一格式，测试样本在[test.json](./datasets/toy_examples/test.json)文件中，这个文件采用json line格式，每行是一个可json化的字符串，json化后"input"字段作为LLM输入，"target"字段为空字符串，待选手们填入模型回复。
    
为完成测试集提交和获得评测得分，参与者需要提供三个文件：
 - 评测参与者需要提交自己训练的LLM在测试集上的生成结果，命名为test_predictions.json文件，保持json line格式。样本数量，样本顺序要与官方提供的test.json文件一致。请大家在提交时注意：每个test样本的答案都需要提交，且不能更改样本"sample_id"字段，否则评测系统将会因为样本数量不对应而无法输出分数。

 - 考虑到各个不同的LLM在回答问题时所采用的格式各不相同，所以在评测中我们仍然需要将回复内容解析为统一的结构化json格式进行提交。结构化的预测结果命名为test_structured.json。 test_structured.json文件整体可以采用json.load()方法加载。其内部结构如下：
 
```bash

{
  "task_name": [
    {
      "sample_id": str,
      "answer": Union[str, list, dict],
    }
  ]
}

```

test_structured.json文件的更具体格式说明见[结构化预测结果格式说明](./src/data/结构化预测结果格式说明.md)。同时，test集与dev集采用相同的结构化格式，请参考[dev集结构化格式](./datasets/toy_examples/dev_structured.json)。


C.参赛者需要提供解析test_predictions.json文件转化为test_structured.json文件的代码，命名为post_generate_process.py 。本代码只限于使用python标准库，并采用如下命令可运行：

```bash
python post_generate_process.py test_predictions.json test_structured.json
```


### 评价指标

本评测任务只有一个测试集，但是其包含多个任务的测试样本，我们采用在各个任务上分别计分的方式进行评测。各个任务上的评测指标如下：

- 对于CMeEE-V2和IMCS-V2-NER任务，采用基于实体实例层面的严格的(strict)，micro的Precision, Recall, F1分数。这里的实体实例包含mention（即实体名称的所有组成字符）和类型这两个组成字段。这里"严格的"指模型必须在指定的样本sample_id上，完全正确预测出ground truth中的实体实例的mention和类型，才能算是成功预测出这个实体实例，则true positive (TP) 加1。而如果模型预测的实体实例不在ground truth中，则false positive (FP)加1。如果ground truth中的实体实例未被模型预测到，则false negative(FN)加1。最终根据整个测试集上的TP，FP，FN计算Precision, Recall, F1分数。
- 对于CMeIE任务，采用基于三元组实例层面的严格的(strict)，micro的precision, recall, F1分数。这里的三元组实例包含头实体mention, 尾实体mention，和关系类型字段。
- 对于CHIP-CDEE任务，采用基于临床事件实例层面的严格的(strict)，micro的precision, recall, F1分数。这里的临床事件实例包含主体词，发生状态，描述词和解剖部位字段。
- 对于IMCS-V2-SR和CHIP-MDCFNPC任务，采用基于临床发现或者症状实例层面的严格的(strict)，micro的precision, recall, F1分数。这里的临床发现或者症状实例包含mention和阴阳性判断标签字段。
- 对CHIP-CDN任务，采用基于ICD-10标准词实例层面的严格的(strict)，micro的precision, recall, F1分数。这里的ICD-10标准词实例包含mention和阴阳性判断标签字段。
- 对CHIP-STS， KUAKE-QQR, KUAKE-IR，KUAKE-QTR任务，我们采用Micro的precision, recall, F1分数作为评估指标。对CHIP-CTC，IMCS-V2-DAC，KUAKE-QIC, 采用Macro的precision, recall, F1分数作为评估指标。
- 对于MedDG和IMCS-V2-MRG数据集，我们采用Rouge-1，Rouge-2，Rouge-L分数作为评估指标。为避免分词影响，计算rouge分数前，会将句子中的汉字拆开，用空格分隔。IMCS-V2-MRG任务中，需要将模型生成的诊断报告拆分为主诉, 现病史, 辅助检查, 既往史, 诊断, 建议这六个章节，分别计算rouge得分后取平均分。

上述任务中，F1(micro/macro)或者Rouge-L将作为每个任务的主要指标。


#### 评测脚本

运行以下命令，选手可在本地对dev集上的模型结果进行评测，并将评测分数保存在scores.json文件中：

```bash

python src/for_evaluation/run_evaluation.py your_path/dev_structured.json your_path/scores.json


```




### 评测规则

- PromptCBLUE的目标是评估LLM在不同医疗任务的总体表现，所以评测参与者只能使用一个LLM模型主干来完成整个测试集的评测。对于测试集中的每一个样本，模型输出必须是一个仅用LLM模型主干上连接的语言模型预测头(LM-head)输出的文本序列，LM-head必须是所有任务共享的。选手的最终模型不能在LM-head以外的其他模块产生与任务直接相关的或是(在模型训练过程中)参与损失计算的logits。

- 如果参与者使用了参数高效微调方法，则其总共使用的参数高效微调模块的总参数量不得超过其LLM模型主干的1%。

- 所有评测参与团队需要在提交测试集时，对其模型训练/微调方法进行介绍，也需要注明其训练数据来源。

- 考虑到目前的LLM训练可能涉及商业数据，大规模模型开源受到各种外在条件的限制，我们将对PromptCBLUE评测开放两个榜单：(1) 通用榜，接受来自企业，高校，开源社区，各类研究团队或者个人进行测试提交评测，不要求其对使用的LLM模型底座开源。但排名靠前的团队需要提供可访问的网页demo地址(最少1天使用权限)供组织者抽查审核。(2) 开源榜，接受各类参赛团队提交评测，但是其必须使用开源的大模型底座，且只能使用开源的数据集进行训练/微调。开源榜排名靠前的团队需要提交模型实现代码至组织者进行审核。

- 评测参与者不得直接使用GPT-4, ChatGPT，文心一言，ChatGLM等公开模型api进行测试集预测(上述模型的研发机构除外)；上述大模型基座可以作为数据增广的来源。

- 评测参与者可以使用任何资源进行LLM训练，包括采用自有的医疗领域(有标注/无标注)数据进行训练。

- 评测参与者可以采用更多的prompt，包括self-instruct方法得到的数据进行评测。但是在预测时，需要采用测试集现有的prompt进行评测。我们正在构建大规模的[医疗领域self-instruct数据集](xxx)。

- 我们鼓励评测参与者采用自有的LLM生成预测结果的同时生成其思维过程。我们目前正在研讨如何对LLM在PromptCBLUE上的思维过程进行人工评测。

- PromptCBLUE在CCKS-2023评测的更多规则详见[CCKS2023-PromptCBLUE天池大赛链接](xxx)


## baseline模型

我们基于[中文医疗大模型ChatMed](https://github.com/michael-wzhu/ChatMed)构建PromptCBLUE的baseline模型。代码和运行操作详见[PromptCBLUE-baseline模型](./src/ft_chatglm_ptuning)。我们考虑以下baseline方法:

- 基于[ChatMed](https://github.com/michael-wzhu/ChatMed)在PromptCBLUE的训练集(68500个样本)上采用p-tuning的参数高效微调方法进行微调；
- 基于ChatGLM-bb模型，在PromptCBLUE的训练集(68500个样本)上采用p-tuning的参数高效微调方法进行微调(bsz=8,gradient accumulation=8, steps=2000)；
- ⏳ TODO: 更多微调方法(如LoRA, Parallel-Adapter等)
- ⏳ TODO: 采用高效微调的方法，针对每个任务进行微调；


在dev集上实验结果如下：

| task         | metric    | ChatMed+ptuning | ChatGLM-6b+ptuning |
|--------------|-----------|--------|--------------------|
| CMeEE-V2     | micro-F1  | 0.6359 | 0.6071             |
| CMeIE        | micro-F1  | 0.3765 | 0.3413             |
| CHIP-CDN     | micro-F1  | 0.7805 | 0.7846             | 
| CHIP-CDEE    | micro-F1  | 0.4914       | 0.4825             | 
| CHIP-STS     | micro-F1  | 0.7696 | 0.6584             |
| CHIP-CTC     | macro-F1  | 0.8046 | 0.7736             |
| KUAKE-IR     | micro-F1  | 0.6154 | 0.5923             |
| KUAKE-QIC    | macro-F1  | 0.8113       | 0.7564             |
| KUAKE-QQR    | micro-F1  | 0.5537       | 0.5899             |
| KUAKE-QTR    | micro-F1  |  0.4701       | 0.4367             |
| CHIP-MDCFNPC | micro-F1  | 0.6865 | 0.6814             |
| IMCS-V2-DAC  | macro-F1  | 0.7059       | 0.7242             |
| IMCS-V2-NER  | micro-F1  | 0.8508 | 0.8527             |
| IMCS-V2-SR   | micro-F1  | 0.6168 | 0.5850             |
| IMCS-V2-MRG  | Rouge-L   | 0.4707 | 0.4765             |
| MedDG        | Rouge-L   | 0.1035       | 0.0938             |
| Overall      | avg score | 0.6090 | 0.5898             |




## 致谢

本项目基于CBLUE基准等数据集进行二次开发，在此对相关项目和研究开发人员表示感谢。

- [CBLUE](https://tianchi.aliyun.com/dataset/95414)

Logo中的小学霸羊驼是由[midjourney](http://midjourney.com)自动生成。



## 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。



## 问题反馈
如有问题，请在GitHub Issue中提交。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 重复以及与本项目无关的issue会被[stable-bot](stale · GitHub Marketplace)处理，敬请谅解。
- 礼貌地提出问题，构建和谐的讨论社区。


## References

- [CBLUE基准](https://tianchi.aliyun.com/dataset/95414)
- [ChatGLM-6b模型](https://github.com/THUDM/ChatGLM-6B)