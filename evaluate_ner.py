# !git clone https://github.com/hhhhh0102/ner_project.git

import numpy as np
import pandas as pd


# 예측된 NER 텍스트 파일, 정답지 NER 텍스트 파일을 input으로 받아 성과지표를 계산하여 csv파일로 만들어주는 함수입니다.
class evaluate:

    def __init__(self):
        PER_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        PER_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        DAT_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        DAT_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        ORG_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        ORG_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        CVL_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        CVL_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        NUM_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        NUM_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        LOC_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        LOC_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        EVT_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        EVT_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        TRM_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        TRM_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        TIM_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        TIM_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        ANM_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        ANM_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        MAT_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        MAT_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        AFW_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        AFW_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        FLD_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        FLD_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        PLT_B_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        PLT_I_DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0,
                     "fall-out": 0,
                     "support": 0}
        _DIC = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "precision": 0, "recall": 0, "accuracy": 0, "f1": 0, "fall-out": 0,
                "support": 0}
        self.tag_dic = {"PER_B": PER_B_DIC, "PER_I": PER_I_DIC, "DAT_B": DAT_B_DIC, "DAT_I": DAT_I_DIC,
                        "ORG_B": ORG_B_DIC,
                        "ORG_I": ORG_I_DIC, "CVL_B": CVL_B_DIC,
                        "CVL_I": CVL_I_DIC, "NUM_B": NUM_B_DIC, "NUM_I": NUM_I_DIC, "LOC_B": LOC_B_DIC,
                        "LOC_I": LOC_I_DIC,
                        "EVT_B": EVT_B_DIC, "EVT_I": EVT_I_DIC,
                        "TRM_B": TRM_B_DIC, "TRM_I": TRM_I_DIC, "TIM_B": TIM_B_DIC, "TIM_I": TIM_I_DIC,
                        "ANM_B": ANM_B_DIC,
                        "ANM_I": ANM_I_DIC, "MAT_B": MAT_B_DIC,
                        "MAT_I": MAT_I_DIC, "AFW_B": AFW_B_DIC, "AFW_I": AFW_I_DIC, "FLD_B": FLD_B_DIC,
                        "FLD_I": FLD_I_DIC,
                        "PLT_B": PLT_B_DIC, "PLT_I": PLT_I_DIC,
                        "-": _DIC}
        # 성과 지표별 macro average, micro average를 계산하기 위한 딕셔너리를 만들고 초기화합니다.
        self.macro_avg = {"precision": 0, "recall": 0, "accuracy": 0, "f1-score": 0, "fall-out": 0}
        self.micro_avg = {"precision": 0, "recall": 0, "accuracy": 0, "f1-score": 0, "fall-out": 0}

    def evaluate_ner(self, predict, actual):
        # 태그별 딕셔너리를 만들고 초기화합니다.

        line_a = []
        line_b = []
        # 각 파일별 tag값들을 읽어옵니다.
        f_a = open(predict, mode='r', encoding='utf-8')
        while True:
            line = f_a.readline()
            if not line: break
            if line == "\n":
                pass
            else:
                line_a.append(line.split()[2])
        f_a.close()

        f_b = open(actual, mode='r', encoding='utf-8')
        while True:
            line = f_b.readline()
            if not line: break
            if line == "\n":
                pass
            else:
                line_b.append(line.split()[2])
        f_b.close()

        # 각 태그별 support, TP, TN, FP, FN값을 계산합니다.

        for l in list(range(len(line_a))):
            predicted_tag = line_a[l]
            actual_tag = line_b[l]
            self.tag_dic[actual_tag]["support"] += 1
            if predicted_tag == actual_tag:
                self.tag_dic[predicted_tag]["TP"] += 1
                for key in self.tag_dic.keys():
                    if key == predicted_tag:
                        pass
                    else:
                        self.tag_dic[key]["TN"] += 1
            elif predicted_tag != actual_tag:
                self.tag_dic[predicted_tag]["FP"] += 1
                for key in self.tag_dic.keys():
                    if key == predicted_tag:
                        pass
                    elif key == actual_tag:
                        self.tag_dic[key]["FN"] += 1
                    else:
                        self.tag_dic[key]["TN"] += 1

        # 태그별 성과 지표를 계산합니다.
        for key in self.tag_dic.keys():
            try:
                self.tag_dic[key]["precision"] += self.tag_dic[key]["TP"] \
                                                  / (self.tag_dic[key]["TP"] + self.tag_dic[key]["FP"])
            except:
                pass
            try:
                self.tag_dic[key]["recall"] += self.tag_dic[key]["TP"] \
                                               / (self.tag_dic[key]["TP"] + self.tag_dic[key]["FN"])
            except:
                pass
            try:
                self.tag_dic[key]["accuracy"] += (self.tag_dic[key]["TP"] + self.tag_dic[key]["TN"]) / (
                        self.tag_dic[key]["TP"] + self.tag_dic[key]["FN"] + self.tag_dic[key]["FP"] +
                        self.tag_dic[key]["TN"])
            except:
                pass
            try:
                self.tag_dic[key]["f1"] += 2 * (self.tag_dic[key]["precision"] * self.tag_dic[key]["recall"]) / (
                        self.tag_dic[key]["precision"] + self.tag_dic[key]["recall"])
            except:
                pass
            try:
                self.tag_dic[key]["fall-out"] += self.tag_dic[key]["FP"] / (
                            self.tag_dic[key]["TN"] + self.tag_dic[key]["FP"])
            except:
                pass

        # CSV파일의 칼럼, 인덱스, 계산된 성과 지표 값들을 지정합니다.

        col = ['precision', 'recall', 'accuracy', 'f1-score', 'fall-out', 'support']
        idx = self.tag_dic.keys()
        value = []
        for key in self.tag_dic.keys():
            if self.tag_dic[key]["support"] == 0:
                value.append([np.nan, np.nan, np.nan, np.nan, np.nan, 0])
            else:
                value.append([round(self.tag_dic[key]["precision"], 4), round(self.tag_dic[key]["recall"], 4),
                              round(self.tag_dic[key]["accuracy"], 4), round(self.tag_dic[key]["f1"], 4),
                              round(self.tag_dic[key]["fall-out"], 4), self.tag_dic[key]["support"]])

        evaluate = pd.DataFrame(value, columns=col, index=idx)
        support_cnt = evaluate["support"].sum()

        # 성과 지표별 macro average와 micro average를 계산합니다.
        for key in self.macro_avg.keys():
            self.macro_avg[key] += round(evaluate[key].mean(), 4)

        for key in self.micro_avg.keys():
            micro_average = 0
            for ix in idx:
                if evaluate["support"][ix] == 0:
                    continue
                micro_average += evaluate[key][ix] * evaluate["support"][ix]
            micro_average /= support_cnt
            self.micro_avg[key] += round(micro_average, 4)

        # 계산된 성과 지표별 macro average와 micro average를 CSV파일에 추가로 구성합니다.

        evaluate.loc['macro average'] = [self.macro_avg["precision"], self.macro_avg["recall"],
                                         self.macro_avg["accuracy"], self.macro_avg["f1-score"],
                                         self.macro_avg["fall-out"], support_cnt]
        evaluate.loc['micro average'] = [self.micro_avg["precision"], self.micro_avg["recall"],
                                         self.micro_avg["accuracy"], self.micro_avg["f1-score"],
                                         self.micro_avg["fall-out"], support_cnt]

        # tag별 성과 지표, 성과 지표들의 macro average와 micro average를 계산한 값들을 csv파일로 만들어줍니다.
        evaluate.to_csv("evaluate.csv", mode='w')


# 예측된 NER 텍스트 파일 : ner_result.txt
# 정답지 NER 텍스트 파일 : ner_train_data.txt
evaluate = evaluate()
evaluate.evaluate_ner("/content/ner_project/ner_result.txt", "/content/ner_project/ner_train_data.txt")
