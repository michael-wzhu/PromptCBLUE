PromptCBLUE


以ChatGPT、GPT-4等为代表的大语言模型（Large Language Model, LLM）掀起了新一轮自然语言处理领域的研究浪潮，展现出了类通用人工智能（AGI）的能力，受到业界广泛关注。在LLM大行其道的背景下，几乎所有的NLP任务都转化为了基于提示的语言生成任务。然而，在中文医学NLP社区中，尚未有一个统一任务形式的评测基准。

为推动LLM在医疗领域的发展和落地，我们现推出**PromptCBLUE**:

- 🚀 将[CBLUE](https://tianchi.aliyun.com/dataset/95414)基准中的各个任务(如医疗文本NER，事件抽取等)转化为基于提示的生成任务; 
- 🚀 为提升LLM的医疗对话能力，增加百万级的在线问诊数据; (TODO)
- 🚀 新增文献标题生成/摘要生成任务；(TODO)
- 🚀 对CBLUE基准中的任务[Text2DT](https://github.com/michael-wzhu/Text2DT_Baseline)进行重构，形成第一个中文医疗推理数据集，用于评测中文LLM在医疗场景下的推理能力；(TODO)