# coding=utf-8

import json
import sys

from evaluators import calc_info_extract_task_scores, calc_cls_task_scores, calc_nlg_task_scores, calc_nlg_task_scores_by_sessions
from src.finetune_bloom_bmt.post_generate_process import process_generated_results


def calc_scores(dict_gt, dict_pred):

    scores = {
        "CMeEE-V2": {},
        "CMeIE": {},
        "CHIP-CDN": {},
        "CHIP-CDEE": {},
        "IMCS-V2-NER": {},
        "CHIP-MDCFNPC": {},
        "IMCS-V2-SR": {},
        "IMCS-V2-DAC": {},
        "CHIP-CTC": {},
        "CHIP-STS": {},
        "KUAKE-IR": {},
        "KUAKE-QIC": {},
        "KUAKE-QQR": {},
        "KUAKE-QTR": {},
        "MedDG": {},
        "IMCS-V2-MRG": {},

    }


    for task_name in scores.keys():

        assert task_name in dict_gt

        if task_name not in dict_pred:
            raise ValueError(f"There are missing predictions in the submission, please check again!")

        gts = dict_gt[task_name]
        preds = dict_pred[task_name]
        if not len(gts) == len(preds):
            raise ValueError(f"There are missing predictions in the submission, please check again!")

        for gt_inst, pred_inst in zip(gts, preds):
            if gt_inst.get("sample_id") != pred_inst.get("sample_id"):
                raise ValueError(
                    f"It seems there are missing predictions or the predicted samples are in the wrong order, "
                    f"please check again! "
                )

        if task_name in ["CMeEE-V2", "CMeIE", "CHIP-CDN",
                         "CHIP-CDEE", "IMCS-V2-NER", "IMCS-V2-SR",
                         "CHIP-MDCFNPC",
                         ]:
            precision, recall, f1 = calc_info_extract_task_scores(
                gts,
                preds
            )
            # print("precision: ", precision)
            # print("recall: ", recall)
            # print("f1: ", f1)

        # "CHIP-STS"
        elif task_name in ["CHIP-STS", ]:
            precision, recall, f1 = calc_cls_task_scores(
                gts,
                preds,
                list_labels=["是的", "不是"],
                return_macro=False,
            )
            # print("CHIP-STS")
            # print("precision: ", precision)
            # print("recall: ", recall)
            # print("f1: ", f1)
            # print("acc: ", acc)

        elif task_name in ["CHIP-CTC", ]:
            precision, recall, f1 = calc_cls_task_scores(
                gts,
                preds,
                list_labels=['非上述类型', '疾病', '症状(患者感受)',
                             '体征(医生检测）', '怀孕相关', '肿瘤进展',
                             '疾病分期', '过敏耐受', '器官组织状态',
                             '预期寿命', '口腔相关', '药物',
                             '治疗或手术', '设备', '护理',
                             '诊断', '实验室检查', '风险评估',
                             '受体状态', '年龄', '特殊病人特征',
                             '读写能力', '性别', '教育情况',
                             '居住情况', '种族', '知情同意',
                             '参与其它试验', '研究者决定', '能力',
                             '伦理审查', '依存性', '成瘾行为',
                             '睡眠', '锻炼', '饮食', '酒精使用',
                             '性取向', '吸烟状况', '献血',
                             '病例来源', '残疾群体', '健康群体',
                             '数据可及性'],
                return_macro=True,
            )
            # print("precision: ", precision)
            # print("recall: ", recall)
            # print("f1: ", f1)
            # print("acc: ", acc)

        elif task_name in ["IMCS-V2-DAC", ]:
            # TODO: 查看样本不均衡性
            list_labels = [
                '非上述类型',
                '关于症状的询问', '关于症状的回答', '关于病因的询问',
                '关于病因的回答', '关于个人基本信息的询问', '关于个人基本信息的回答',
                '关于已有检查和治疗的提问', '关于已有检查和治疗的提问', '关于用药建议的提问',
                '关于用药建议的解答', '关于就医建议的提问', '关于就医建议的解答',
                '关于注意事项的提问', '关于注意事项的解答', '诊断'
            ]
            precision, recall, f1 = calc_cls_task_scores(
                gts,
                preds,
                list_labels=list_labels,
                return_macro=True,
            )
            # print("precision: ", precision)
            # print("recall: ", recall)
            # print("f1: ", f1)
            # print("acc: ", acc)



        elif task_name in ["KUAKE-IR", ]:
            list_labels = ["相关", "不相关"]
            precision, recall, f1 = calc_cls_task_scores(
                gts,
                preds,
                list_labels=list_labels,
                return_macro=False,
            )

        elif task_name in ["KUAKE-QIC", ]:
            list_labels = [
                '非上述类型',
                "病情诊断",
                "病因分析",
                "治疗方案",
                "就医建议",
                "指标解读",
                "疾病描述",
                "后果表述",
                "注意事项",
                "功效作用",
                "医疗费用",
            ]
            precision, recall, f1 = calc_cls_task_scores(
                gts,
                preds,
                list_labels=list_labels,
                return_macro=True,
            )

        elif task_name in ["KUAKE-QTR", ]:
            list_labels = [
                "完全不匹配",
                "很少匹配",
                "部分匹配",
                "完全匹配",
            ]
            precision, recall, f1 = calc_cls_task_scores(
                gts,
                preds,
                list_labels=list_labels,
                return_macro=False,
            )

        elif task_name in ["KUAKE-QQR", ]:
            list_labels = [
                "完全一致",
                "后者是前者的语义子集",
                "后者是前者的语义父集或语义毫无关联",
            ]
            precision, recall, f1 = calc_cls_task_scores(
                gts,
                preds,
                list_labels=list_labels,
                return_macro=False,
            )

        elif task_name in [
            "MedDG",
        ]:
            rouge1, rouge2, rougeL = calc_nlg_task_scores(
                gts,
                preds,
            )
            print("rouge1: ", rouge1)
            print("rouge2: ", rouge2)
            print("rougeL: ", rougeL)

        elif task_name in [
            "IMCS-V2-MRG",
        ]:
            rouge1, rouge2, rougeL = calc_nlg_task_scores_by_sessions(
                gts,
                preds,
            )
            print("rouge1: ", rouge1)
            print("rouge2: ", rouge2)
            print("rougeL: ", rougeL)


if __name__ == "__main__":
    # prediction_file = "medical_prompts/src/for_evaluation/test_pred.json"

    gt_file_structured = "medical_prompts/datasets/prompt_datasets/multitask/test_structured.json"
    prediction_file = sys.argv[1]

    dict_pred = process_generated_results(prediction_file)

    dict_gt = json.load(
        open(gt_file_structured, "r", encoding="utf-8")
    )

    try:
        calc_scores(dict_gt, dict_pred)
    except Exception as e:
        print(e)




