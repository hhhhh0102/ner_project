import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tokenizer import KoBertTokenizer
from sklearn.model_selection import train_test_split
from Preprocess import preprocess

class make_input:
    def __init__(self):
        self.max_len = 88
        self.bs = 32
        self.pr = preprocess()
        self.tr_inputs = None
        self.val_inputs = None
        self.tr_tags = None
        self.val_tags = None
        self.tr_masks = None
        self.val_masks = None
        self.resolver = None
        self.dataset = "train_data"

    def set_dataset(self, dataset):
        self.dataset = dataset

    def make_input(self):
        self.pr.preprocessing(self.dataset)
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        tokenized_texts_and_labels = [
            tokenizer.tokenize_and_preserve_labels(sent, labs)
            for sent, labs in zip(self.pr.sentences, self.pr.targets)]
        tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
        labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=self.max_len, dtype="int", value=tokenizer.convert_tokens_to_ids("[PAD]"),
                                  truncating="post", padding="post")

        tags = pad_sequences([lab for lab in labels], maxlen=self.max_len, value=self.pr.label_dict["[PAD]"],
                             padding='post', \
                             dtype='int', truncating='post')

        attention_masks = np.array(
            [[int(i != tokenizer.convert_tokens_to_ids("[PAD]")) for i in ii] for ii in input_ids])

        self.tr_inputs, self.val_inputs, self.tr_tags, self.val_tags = train_test_split(input_ids, tags,
                                                                                        random_state=2018,
                                                                                        test_size=0.1)

        self.tr_masks, self.val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                               random_state=2018, test_size=0.1)