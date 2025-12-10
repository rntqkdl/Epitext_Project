# 3_model - 모델 학습/평가

NLP 및 Vision 모델 학습과 평가를 관리하는 모듈입니다.

## 구조

- `nlp/`
  - `sikuroberta/`: 한자 인식 모델 (SikuRoBERTa)
  - `gemini_experiment/`: Gemini 기반 평가
- `vision/`
  - `swin_experiment/`: Vision Transformer (SwinV2)
  - `ocr_experiment/`: OCR 모델 평가
- `saved_models/`: 학습된 모델 저장 경로

## 실행 예시

모든 명령은 `Epitext_Project` 루트에서 실행합니다.

### 개별 모델 학습

```bash
# SikuRoBERTa 학습
python 3_model/main.py --task sikuroberta_train

# SwinV2 학습
python 3_model/main.py --task swin_train
```

### 개별 모델 평가

```bash
# SikuRoBERTa 평가
python 3_model/main.py --task sikuroberta_eval

# Gemini 평가
python 3_model/main.py --task gemini_eval

# SwinV2 평가
python 3_model/main.py --task swin_eval

# OCR 평가
python 3_model/main.py --task ocr_eval
```

### 일괄 실행

```bash
# 모든 모델 학습
python 3_model/main.py --task all_train

# 모든 모델 평가
python 3_model/main.py --task all_eval
```

### 상세 로그 출력

```bash
python 3_model/main.py --task sikuroberta_train -v
```

## 설정

`config.py`에서 다음을 설정할 수 있습니다:

- `RANDOM_SEED`: 재현성 보장 (기본값: 42)
- `DEVICE`: "cuda" 또는 "cpu"
- 각 모델의 저장 경로

## 참고사항

- 모델 학습 코드는 각 폴더의 `train/main.py`에 위치합니다.
- 평가 코드는 각 폴더의 `evaluation/main.py`에 위치합니다.
