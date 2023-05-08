

###  数据

请前往[PromptCBLUE通用赛道评测网站](https://tianchi.aliyun.com/competition/entrance/532085/introduction)或者[PromptCBLUE通用赛道评测网站](https://tianchi.aliyun.com/competition/entrance/532084/introduction)下载训练集，验证集以及测试集A或者测试集B。这些数据放置在自己指定的文件夹中，如"datasets/PromptCBLUE/toy_examples"。


### 训练

```bash
./src/ft_chatglm_ptuning/train.sh

```


### 预测(生成回复)

```bash
./src/ft_chatglm_ptuning/evaluate.sh

```