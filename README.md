# ner_project
링크인턴 얼리버드프로젝트 과제

# ner_task.ipynb
NER 예측 모듈

# evaluate.ipynb
NER 모듈의 성과지표를 계산하고, 계산된 값을 저장하는 모듈

# pre_trained_data.txt
NER값을 계산 하려는 한글 텍스트 파일

# ner_result.txt
예측된 NER값 파일

# ner_train_data.txt
NER 정답지 파일

# ner_model
bert를 이용한 ner 예측 모델을 만들기 위한 라이브러리 입니다
https://github.com/kimwoonggon/publicservant_AI/blob/master/5_(BERT_실습)한국어_개체명_인식.ipynb 를 참조하였습니다.

## Preprocess.py
dataset의 전처리를 하는 모듈입니다

## tokenizer.py
한글 토크나이징을 위한 모듈입니다

## make_input.py
bert 모델을 사용하기 위해 input의 사이즈를 조정하는 모듈입니다

## make_model.py
NER 예측 모델을 만드는 모듈입니다
