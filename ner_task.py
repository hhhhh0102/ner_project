import numpy as np
import tensorflow as tf
from tokenizer import KoBertTokenizer


def ner_task(textpath):
    tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    model = tf.keras.models.load_model("my_model.h5")
    line_list = []
    index_to_ner = {0: 'PER_B', 1: 'DAT_B', 2: '-', 3: 'ORG_B', 4: 'CVL_B', 5: 'NUM_B',
                    6: 'LOC_B', 7: 'EVT_B', 8: 'TRM_B', 9: 'TRM_I', 10: 'EVT_I', 11: 'PER_I',
                    12: 'CVL_I', 13: 'NUM_I', 14: 'TIM_B', 15: 'TIM_I', 16: 'ORG_I', 17: 'DAT_I',
                    18: 'ANM_B', 19: 'MAT_B', 20: 'MAT_I', 21: 'AFW_B', 22: 'FLD_B', 23: 'LOC_I',
                    24: 'AFW_I', 25: 'PLT_B', 26: 'FLD_I', 27: 'ANM_I', 28: 'PLT_I', 29: '[PAD]'}

    f = open(textpath, mode='r', encoding='utf-8')
    while True:
        line = f.readline()
        if not line: break
        line_list.append(line)
    f.close()

    f = open('ner_result.txt', mode='wt', encoding='utf-8')
    for l in list(range(len(line_list))):

        tokenized_sentence = np.array([tokenizer.encode(line_list[l], max_length=88, pad_to_max_length=True)])
        tokenized_mask = np.array([[int(x != 1) for x in tokenized_sentence[0].tolist()]])
        ans = model.predict([tokenized_sentence, tokenized_mask])
        ans = np.argmax(ans, axis=2)

        tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, ans[0]):

            if (token.startswith("▁")):
                new_labels.append(index_to_ner[label_idx])
                new_tokens.append(token[1:])
            elif (token == '[CLS]'):
                pass
            elif (token == '[SEP]'):
                pass
            elif (token == '[PAD]'):
                pass
            elif (token != '[CLS]' or token != '[SEP]'):
                new_tokens[-1] = new_tokens[-1] + token

        for ll in list(range(len(new_tokens))):
            f.write("{}\t{}\t{}\n".format(ll + 1, new_tokens[ll], new_labels[ll]))
        f.write("\n")

    f.close()

ner_task("pre_trained_data.txt")