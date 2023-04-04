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

| Query       | Answer |
|-------------|--------|
| 版本号         | v0.1   |
| prompt 模板数量 | xxx    |
| 训练集         | xxx    |
| 验证集         | xxx    |
| 测试集         | ⏳      |



### 数据统计


| 数据集      | #Train | #dev | #test | prompt模板数量 | 评价指标 | 
|----------|--------|------|-------|-----------|------|
| CMeEE-v2 | -      | -    | -     | -         | -    |



### 数据集构建

经过改造后，医疗文本NLP数据集都将转化为如下格式。input字段是模型的输入，target字段是模型的输出，type是原任务类型(不作为模型输入)，answer_choices字段是选项，只有分类、推理类任务有此字段。

```bash
{
  "input": str,
  "target": str,
  "type": str,
  "answer_choices": str,
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