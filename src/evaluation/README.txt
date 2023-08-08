
## 验证：LLM回复转化为结构化格式的代码（参赛选手需要根据自己的LLM输出格式编写格式转化代码）：
python post_generate_process.py dev_predictions.json results.json

## 评分
./py_entrance.sh input_param.json eval_result.json
cat eval_result.json

或者运行
python evaluate.py input_param.json eval_result.json
cat eval_result.json

## 参考
更多详细的数据说明和baseline方法实现，见https://github.com/michael-wzhu/PromptCBLUE