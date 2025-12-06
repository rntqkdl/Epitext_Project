# SikuRoBERTa MLM 학습 및 평가

## 개요
탁본 한문 데이터로 SikuRoBERTa를 Fine-tuning하여 문맥 분석 성능을 향상시킵니다.

## 실행 방법

### 1. 환경 설정
```bash
pip install -U transformers datasets tokenizers accelerate pandas tensorboard tqdm torch matplotlib
2. 학습
bash
cd 3_model/nlp/sikuroberta
python train.py
특징:

절대경로 자동 설정 (어디서든 실행 가능)

체크포인트 자동 재개

Early Stopping

TensorBoard 로깅

3. 평가
bash
python evaluate.py
출력:

Test Loss, Perplexity

Top-1, Top-5 Accuracy

결과 파일 자동 저장

4. TensorBoard
bash
tensorboard --logdir=logs/tensorboard
성능 (목표)
Top-5 Accuracy: 90.3%

Perplexity: < 10.0

파일 구조
text
3_model/nlp/sikuroberta/
├── config.py          # 설정 (경로 자동 생성)
├── train.py           # 학습 코드
├── evaluate.py        # 평가 코드
├── README.md          # 이 파일
├── checkpoints/       # 체크포인트 (자동 생성)
├── final_model/       # 최종 모델 (자동 생성)
└── logs/              # 로그 및 그래프 (자동 생성)
    ├── tensorboard/
    ├── loss_graph.png
    └── test_results.txt

주요 파라미터 (config.py)
학습
배치 사이즈: 4

Gradient Accumulation: 8 (유효 배치: 32)

학습률: 2e-5

Epochs: 10

MLM 확률: 15%

Early Stopping: 3 epochs

평가
배치 사이즈: 8

주의사항
전처리 필수: 먼저 1_data/에서 데이터 전처리를 실행하세요

GPU 메모리: RTX 2060 6GB 기준 (배치 사이즈 조정 가능)

절대경로: config.py에서 자동으로 설정됩니다

출처 및 참고
베이스 모델: SIKU-BERT/sikuroberta (Hugging Face)

학습/평가 코드: 팀 자체 작성

프로젝트: 4조 복원왕 김탁본 (2025)

참고 논문
Song et al. (2025). Qwen Series for Ancient Chinese Processing

Kim et al. (2024). Korean-Chinese Classical Text Processing
