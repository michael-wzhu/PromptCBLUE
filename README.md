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

为推动LLM在医疗领域的发展和落地，我们现推出**PromptCBLUE**:

- 🚀 将[CBLUE](https://tianchi.aliyun.com/dataset/95414)基准中的各个任务(如医疗文本NER，事件抽取等)转化为基于提示的生成任务; 
- 🚀 为提升LLM的医疗对话能力，增加百万级的在线问诊数据; (TODO)
- 🚀 新增文献标题生成/摘要生成任务；(TODO)
- 🚀 对CBLUE基准中的任务[Text2DT](https://github.com/michael-wzhu/Text2DT_Baseline)进行重构，形成第一个中文医疗推理数据集，用于评测中文LLM在医疗场景下的推理能力；(TODO)


----

[Text2DT](https://github.com/michael-wzhu/Text2DT_Baseline) | [中文医疗大模型ChatMed](https://github.com/michael-wzhu/ChatMed)


## 更新

2023/4/5 上传PromptCBLUE的v0.1版本，包含xxx个任务共xxx条数据. 数据下载: [百度网盘(提取码: xxx)]()

2023/4/3 PromptCBLUE上线了，将持续更新(目前版本只包含train/valid split，后续将提供在线评估平台) 🎉🎉🎉




## 数据集详情

### PromptCBLUE总体统计


| 微调训练场景      | -    |
|-------------|------|
| 版本号         | v0.1 |
| prompt 模板数量 | xxx  |
| 训练集         | xxx  |
| 验证集         | xxx  |
| 测试集         | ⏳    |


### 数据统计


| 数据集      | #Train | #dev | #test | prompt模板数量 | 评价指标 | 
|----------|--------|------|-------|-----------|------|
| CMeEE-v2 | -      | -    | -     | -         | -    |



### 数据集构建

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


### 任务改造

为了将CBLUE中的各种不同任务适配为符合LLM的输入输出格式，我们对CBLUE各个数据集进行如下改造：

#### CMeEE任务

本任务原本是标准的医学文本NER任务，选手需要给出医学实体mention在待抽取文本中的具体span位置。在PromptCBLUE中，本任务被改造为：根据指定的实体类型，生成实体mention。在评分时，我们只考虑实体mention及其类型标签，不再考虑span位置信息。样例如下:

```bash

{
  "input": "医学实体识别：\n外周血白细胞计数常明显升高，伴核左移。\n实体选项：疾病，医学检验项目，医院科室，身体部位，微生物类，临床表现，药物\n答：", 
  "target": "上述句子中的实体包含：\n医学检验项目实体：外周血白细胞计数\n疾病实体：\n医院科室实体：\n药物实体：\n微生物类实体：", 
  "answer_choices": ["疾病", "医学检验项目", "医院科室", "身体部位", "微生物类", "临床表现", "药物"], 
  "task_type": "ner", 
  "task_dataset": "CMeEE-V2", 
  "sample_id": "train-134372"
}

```

#### CMeIE任务

本任务是三元组联合抽取任务。在PromptCBLUE中，我们将其定义为：在指定的关系类型下，抽取形成该关系的头尾实体mention。参赛者可以根据需要对本任务的指令/提示进行进一步拆解，以更好的完成任务。示例如下：

```bash

{
  "input": "找出句子中的具有临床表现，同义词关系类型的头尾实体对：\n妊娠期高血压@健康孕妇可以通过全身血管阻力 (systemic vascular resistance，SVR) 和心输出量下降得以平衡，而妊娠期高血压妇女SVR较低，心输出量往往会增加。", 
  "target": "上述句子中临床表现关系的实体对如下：\n头实体：妊娠期高血压，尾实体：SVR较低；头实体：妊娠期高血压，尾实体：心输出量往往会增加；\n上述句子中同义词关系的实体对如下：", 
  "answer_choices": ["临床表现", "同义词"], 
  "task_type": "spo_generation", 
  "task_dataset": "CMeIE", 
  "sample_id": "train-67405"
}

```


#### CDEE任务

本任务是医学事件抽取，即给定一段现病史或者医学影像所见报告，要求从中抽取临床发现事件的四个属性: 解剖部位、主体词、描述词，以及发生状态。在PromptCBLUE下，该任务表现为根据给定文本，生成相应的事件信息描述。注意原任务的标注中，部分主体词是做了ICD10标准化的，我们将这一部分样本过滤掉了，也就是说我们目前主要是评估LLM的医学事件抽取能力。

```bash

{
  "input": "7月前患者给予亚砷酸氯化钠(伊泰达)注射液 10mg 静滴14天，6月前予以口服维甲酸 20mg bid*14天维持治疗，5月前行亚砷酸氯化钠(伊泰达)注射液 10mg 静滴14天维持化疗，3月余前复查骨髓检查示增生性骨髓象；fish：pml/rara（双色双融合）(15/17)：未见异常；腰穿脑脊液未见异常细胞。现为维持化疗入院。\n问题：句子中的临床发现事件及其属性是什么？\n说明：临床发现事件由主体词，发生状态，描述词和解剖部位组成", 
  "target": "上述句子中的临床发现事件如下：\n主体词：fish：pml/rara（双色双融合）(15/17)异常；发生状态：否定；描述词：；解剖部位：\n主体词：骨髓象；发生状态：；描述词：增生性；解剖部位：骨髓", 
  "answer_choices": null, 
  "task_type": "event_extraction", 
  "task_dataset": "CHIP-CDEE", 
  "sample_id": "train-17503"
}

```



#### CDN任务

该任务原型是给定一诊断原词，要求给出其对应的诊断标准词，而诊断标准词是从ICD-10这一个4w+的标准词库中选择。由于我们不可能一次性将四万个词输入到LLM中，所以我们将CDN任务改造为：给定原词，从候选的若干个ICD-10诊断标准词中选择出匹配的词(可能有多个, 可能一个都没有)。

```bash

{
  "input": "主动脉弓缩窄心功能低下\n归一化后的标准词是？\n实体选项：胫前动脉假性动脉瘤，主动脉缩窄，男性性腺功能低下，男性性腺功能低下，垂体功能低下，心功能不全\n说明：从候选的若干个ICD-10诊断标准词中选择出与原诊断描述匹配的词\n答：", 
  "target": "主动脉缩窄，心功能不全", 
  "answer_choices": ["胫前动脉假性动脉瘤", "主动脉缩窄", "男性性腺功能低下", "男性性腺功能低下", "垂体功能低下", "心功能不全"], 
  "task_type": "normalization", 
  "task_dataset": "CHIP-CDN", 
  "sample_id": "train-17932"
}



```






### 数据样例



## baseline模型

TODO: 我们基于[中文医疗大模型ChatMed](https://github.com/michael-wzhu/ChatMed)构建PromptCBLUE的baseline模型。同时我们也评估了现有的开源中文LLM，如Chinese-LLaMA.

### 零样本设定

在此设定下，所有的模型都只经过预训练，而未在具体NLP任务上进行微调。

实验结果：⏳

### 少样本设定

在此设定下，百亿以内模型在每个数据集1000个训练样本的设定下进行高效参数微调。

实验结果：⏳

### 全样本设定

在此设定下，百亿以内模型在每个数据集全量训练样本的设定下进行高效参数微调或者全量微调。

实验结果：⏳



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