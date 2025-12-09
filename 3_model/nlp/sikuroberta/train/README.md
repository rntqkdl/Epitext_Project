
# SikuRoBERTa 학습 모듈 (Training)



한자 탁본 텍스트 데이터를 사용하여 MLM(Masked Language Modeling) 학습을 수행하는 모듈입니다.



## 1. 파일 구성

```text

train/

├── __init__.py      # 패키지 초기화

├── config.py        # 학습 전용 설정 (경로, 하이퍼파라미터)

├── train_task.py    # 메인 학습 스크립트

└── README.md        # 설명서

2. 사용 방법

설정 변경

config.py 파일에서 다음 항목을 수정할 수 있습니다.



DATASET_PATH: 전처리된 데이터셋 경로 (HuggingFace Dataset 포맷)

EPOCHS: 학습 반복 횟수

BATCH_SIZE: 배치 크기

LEARNING_RATE: 학습률

실행 방법

터미널에서 아래 명령어로 학습을 시작합니다.



Bash



cd 3_model/nlp/sikuroberta/train

python train_task.py

3. 출력 결과

체크포인트: saved_models/sikuroberta/checkpoints/ (학습 중간 저장)

최종 모델: saved_models/sikuroberta/final/ (학습 완료 후 저장)

로그: logs/sikuroberta/loss_graph.png (학습 손실 그래프)
