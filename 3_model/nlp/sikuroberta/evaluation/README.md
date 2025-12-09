
# SikuRoBERTa 검증 모듈 (Evaluation)



학습된 SikuRoBERTa 모델을 로드하여 Test 데이터셋에 대한 정량적 성능 지표(Loss, Perplexity, Accuracy)를 산출하는 모듈입니다.



## 1. 파일 구성

```text

evaluation/

├── __init__.py      # 패키지 초기화

├── config.py        # 검증 전용 설정 (경로, 파라미터)

├── evaluate_task.py # 메인 검증 스크립트

└── README.md        # 설명서

2. 사용 방법

설정 변경

config.py 파일에서 다음 항목을 수정할 수 있습니다.



MODEL_PATH: 평가할 학습된 모델 경로 (기본값: saved_models/sikuroberta/final)

TEST_DATA_PATH: 테스트 데이터셋 경로

실행 방법

터미널에서 아래 명령어로 평가를 시작합니다.







cd 3_model/nlp/sikuroberta/evaluation

python evaluate_task.py

3. 출력 결과

콘솔 출력: Loss, Perplexity, Top-1 Accuracy, Top-5 Accuracy

결과 파일: logs/sikuroberta/eval_results.txt (평가 결과 요약 저장)
