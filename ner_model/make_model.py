import tensorflow as tf
import os
from make_input import make_input
from transformers import *


class make_model:
    def __init__(self):
        self.model_input = make_input()
        try:
            self.resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        except:
            pass

    def create_model(self):
        # TPU 작동을 위해 실행
        self.resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(self.resolver)
        tf.tpu.experimental.initialize_tpu_system(self.resolver)

        SEQ_LEN = self.model_input.max_len
        model = TFBertModel.from_pretrained("monologg/kobert", from_pt=True,
                                            num_labels=len(self.model_input.pr.label_dict),
                                            output_attentions=False,
                                            output_hidden_states=False)
        token_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_word_ids')  # 토큰 인풋
        mask_inputs = tf.keras.layers.Input((SEQ_LEN,), dtype=tf.int32, name='input_masks')  # 마스크 인풋

        bert_outputs = model.bert([token_inputs, mask_inputs])
        bert_outputs = bert_outputs[0]  # shape : (Batch_size, max_len, 30(개체의 총 개수))
        nr = tf.keras.layers.Dense(30, activation='softmax')(bert_outputs)  # shape : (Batch_size, max_len, 30)

        nr_model = tf.keras.Model([token_inputs, mask_inputs], nr)

        nr_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00002),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                         metrics=['sparse_categorical_accuracy'])
        nr_model.summary()
        return nr_model

    def train_model(self):
        self.model_input.make_input()
        tf.config.experimental_connect_to_cluster(self.resolver)
        tf.tpu.experimental.initialize_tpu_system(self.resolver)
        strategy = tf.distribute.experimental.TPUStrategy(self.resolver)
        # TPU를 활용하기 위해 context로 묶어주기
        with strategy.scope():
            nr_model = self.create_model()
            nr_model.fit([self.model_input.tr_inputs, self.model_input.tr_masks], self.model_input.tr_tags,
                         validation_data=([self.model_input.val_inputs, self.model_input.val_masks],
                                          self.model_input.val_tags), epochs=10, shuffle=False,
                         batch_size=self.model_input.bs)
            nr_model.save("my_model.h5")

    def load_model(self):
        model = tf.keras.models.load_model("my_model.h5")
        return model
