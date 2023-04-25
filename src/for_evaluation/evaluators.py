import time

import evaluate
import numpy as np
import sklearn.metrics
from sklearn.metrics import classification_report
from transformers import BasicTokenizer

basic_tokenizer = BasicTokenizer(tokenize_chinese_chars=True)


def calc_info_extract_task_scores(list_structured_golden,
                                  list_structured_predict):

    assert len(list_structured_golden) == len(list_structured_predict)

    tp = 0
    fp = 0
    fn = 0
    for samp_golden, samp_predict in zip(list_structured_golden, list_structured_predict):
        assert samp_golden["sample_id"] == samp_predict["sample_id"]
        answer_golden = samp_golden["answer"]
        answer_predict = samp_predict["answer"]

        assert isinstance(answer_golden, list)
        assert isinstance(answer_predict, list)

        set_golden = set()
        for inst in answer_golden:
            assert isinstance(inst, dict)
            keys = sorted(list(inst.keys()))
            inst = tuple(["，".join(inst[w]) if isinstance(inst[w], list) else inst[w] for w in keys ])
            # inst = list(inst.items())
            # inst.sort()
            # inst = tuple(inst)

            set_golden.add(inst)

        set_predict = set()
        for inst in answer_predict:
            assert isinstance(inst, dict)
            keys = sorted(list(inst.keys()))
            # inst = tuple([inst[w] for w in keys])
            inst = tuple(["，".join(inst[w]) if isinstance(inst[w], list) else inst[w] for w in keys])

            # inst = list(inst.items())
            # inst.sort()
            # inst = tuple(inst)

            set_predict.add(inst)

        # print("set_predict: ", set_predict)
        # print("set_golden: ", set_golden)

        tp += len(set_golden.intersection(set_predict))
        fp += len(set_predict.difference(set_golden))
        fn += len(set_golden.difference(set_predict))

    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    return precision, recall, f1


def calc_cls_task_scores(list_structured_golden,
                         list_structured_predict,
                         list_labels=None,
                         return_macro=False,
                         ):
    # types = list_labels
    # scores = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in list_labels + ["ALL"]}

    predictions = []
    ground_truths = []

    # Count GT relations and Predicted relations
    assert len(list_structured_golden) == len(list_structured_predict)
    n_sents = len(list_structured_golden)

    # Count TP, FP and FN per type
    for pred_samp, gt_samp in zip(list_structured_predict, list_structured_golden):
        assert pred_samp["sample_id"] == gt_samp["sample_id"]

        pred_label = pred_samp["answer"]
        gt_label = gt_samp["answer"]

        predictions.append(pred_label)
        ground_truths.append(gt_label)

    # metric
    t0 = time.time()
    cls_report = classification_report(
        ground_truths, predictions,
        output_dict=True
    )
    print(cls_report)

    t1 = time.time()
    print("calculation metrics: ", t1 - t0)

    if return_macro:
        return cls_report["macro avg"]["precision"], \
               cls_report["macro avg"]["recall"], \
               cls_report["macro avg"]["f1-score"]
    else:
        return cls_report["weighted avg"]["precision"], \
               cls_report["weighted avg"]["recall"], \
               cls_report["weighted avg"]["f1-score"]


def calc_nlg_task_scores(list_structured_golden, list_structured_predict):


    assert len(list_structured_golden) == len(list_structured_predict)

    scores = []
    predictions = []
    references = []
    for samp_golden, samp_predict in zip(list_structured_golden, list_structured_predict):
        # print("samp_golden: ", samp_golden)
        # print("samp_predict: ", samp_predict)

        assert samp_golden["sample_id"] == samp_predict["sample_id"]
        answer_golden = samp_golden["answer"]
        answer_predict = samp_predict["answer"]

        assert isinstance(answer_golden, str)
        assert isinstance(answer_predict, str)

        # basic tokenizer: 拆分中文字，保留英文单词
        answer_predict = basic_tokenizer.tokenize(answer_predict)
        answer_golden = basic_tokenizer.tokenize(answer_golden)
        answer_predict = " ".join(answer_predict).strip()
        answer_golden = " ".join(answer_golden).strip()
        if answer_golden.strip() == "":
            answer_golden = "无 。"
        if answer_predict.strip() == "":
            answer_predict = "无 。"
        print("answer_predict: ", answer_predict)
        print("answer_golden: ", answer_golden)

        predictions.append(answer_predict)
        references.append(answer_golden)

    from rouge_chinese import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)

    rouge1 = scores["rouge-1"]["f"]
    rouge2 = scores["rouge-2"]["f"]
    rougeL = scores["rouge-l"]["f"]

    return rouge1, rouge2, rougeL


def calc_nlg_task_scores_by_sessions(list_structured_golden, list_structured_predict):


    assert len(list_structured_golden) == len(list_structured_predict)

    scores = []
    predictions = []
    references = []
    for samp_golden, samp_predict in zip(list_structured_golden, list_structured_predict):
        # print("samp_golden: ", samp_golden)
        # print("samp_predict: ", samp_predict)

        assert samp_golden["sample_id"] == samp_predict["sample_id"]
        answer_golden = samp_golden["answer"]
        answer_predict = samp_predict["answer"]

        assert list(answer_golden.keys()) == len(answer_predict.keys())

        for key in answer_golden.keys():
            pred = answer_predict[key].strip()
            gt = answer_golden[key].strip()

            # basic tokenizer: 拆分中文字，保留英文单词
            pred = basic_tokenizer.tokenize(pred)
            gt = basic_tokenizer.tokenize(gt)
            pred = " ".join(pred).strip()
            gt = " ".join(gt).strip()
            if gt.strip() == "":
                gt = "无 。"
            if pred.strip() == "":
                pred = "无 。"

            predictions.append(
                pred
            )
            references.append(
                gt
            )

    from rouge_chinese import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)

    rouge1 = scores["rouge-1"]["f"]
    rouge2 = scores["rouge-2"]["f"]
    rougeL = scores["rouge-l"]["f"]

    return rouge1, rouge2, rougeL



