# SikuRoBERTa MLM 학습 및 평가

## 📌 개요
탁본 한문 데이터로 SikuRoBERTa를 Fine-tuning하여 문맥 분석 성능을 향상시킵니다.

## 🚀 실행 방법

### 1. 환경 설정
```bash
pip install -U transformers datasets tokenizers accelerate pandas tensorboard tqdm

# ============================================================
# STEP 15-3: README 생성
# ============================================================

Write-Host "
📝 STEP 15-3: README 생성..." -ForegroundColor Cyan

@"
# SikuRoBERTa MLM 학습 및 평가

## 📌 개요
탁본 한문 데이터로 SikuRoBERTa를 Fine-tuning하여 문맥 분석 성능을 향상시킵니다.

## 🚀 실행 방법

### 1. 환경 설정
```bash
pip install -U transformers datasets tokenizers accelerate pandas tensorboard tqdm
2. 학습
bash
python train.py
3. 평가
bash
python evaluate.py
4. TensorBoard
bash
tensorboard --logdir=C:\Users\김선민\Downloads\punmodel\runs_simple_128
📊 성능
Top-1 Accuracy: XX.XX%

Top-5 Accuracy: 90.3%

Perplexity: XX.XX

📁 파일 구조
text
3_model/nlp/sikuroberta/
├── train.py           # 학습 코드
├── evaluate.py        # 평가 코드
└── README.md          # 설명서
🔧 주요 파라미터
배치 사이즈: 4 (Gradient Accumulation: 8)

학습률: 2e-5

Epochs: 10

MLM 확률: 15%

📚 출처
Backend: https://github.com/jae2022/Epitext_Back

팀: 4조 복원왕 김탁본 (2025)
