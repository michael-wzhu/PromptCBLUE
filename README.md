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


## 致谢

本项目基于CBLUE基准等数据集进行二次开发，在此对相关项目和研究开发人员表示感谢。

- [CBLUE](https://tianchi.aliyun.com/dataset/95414)


## 数据集详情

### PromptCBLUE总体统计

| :----------------- | :------: |
| 版本号 | v0.1 |
| prompt模板数量 | xxx |
| 训练集 | xxx |
| 验证集 | xxx |
| 测试集 | ⏳  |



## 免责声明

本项目相关资源仅供学术研究之用，严禁用于商业用途。


## 问题反馈
如有问题，请在GitHub Issue中提交。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 重复以及与本项目无关的issue会被[stable-bot](stale · GitHub Marketplace)处理，敬请谅解。
- 礼貌地提出问题，构建和谐的讨论社区。