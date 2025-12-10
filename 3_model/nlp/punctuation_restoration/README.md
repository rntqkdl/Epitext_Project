# 구두점 복원 모델

이 모듈은 한문 텍스트에서 누락된 구두점을 자동으로 복원합니다. SikuRoBERTa
토큰 분류 모델을 사용하여 문자별로 구두점을 예측하고, 슬라이딩 윈도우
방식을 적용하여 긴 텍스트도 효율적으로 처리합니다. 전체 데이터셋을
처리하려면 `main.py`를 실행하고, 단일 문자열을 복원하려면
`restore_punctuation_sliding()` 함수를 이용하세요.

## 파일 구성

- **config.py** – 모델 태그, 입력/출력 경로, 윈도우 크기 등 설정을 정의합니다.
- **main.py** – CSV 파일을 읽어 구두점을 복원하고 결과를 저장하는 실행 스크립트입니다.
- **\_\_init\_\_.py** – 패키지 초기화 모듈로, 사용 예시를 제공합니다.

## 사용법

### 1. 스크립트 실행

프로젝트 루트에서 다음 명령을 실행하여 데이터셋 전체의 구두점을 복원할 수 있습니다.

```bash
cd 3_model/nlp/punctuation_restoration
python main.py
```

실행 결과는 `1_data/raw_data/doc_id_transcript_with_punctuation.csv`에 저장됩니다.

### 2. 모듈 임포트 및 함수 사용

다른 파이썬 코드에서 구두점 복원 기능을 사용하려면 다음과 같이 임포트할 수 있습니다.

```python
from epitext_model.nlp.punctuation_restoration.main import (
    download_model,
    load_model,
    remove_punctuation,
    restore_punctuation_sliding,
)
from epitext_model.nlp.punctuation_restoration import config

# 모델 다운로드 및 로드
download_model(config.MODEL_TAG, config.MODEL_CACHE_DIR)
model_info = load_model(config.MODEL_CACHE_DIR, device="cpu")

# 단일 문장 처리 예시
cleaned = remove_punctuation("예시 문장입니다")
restored = restore_punctuation_sliding(cleaned, model_info)
print(restored)
```

`epitext_model` 패키지는 `setup_import_aliases.py`를 실행하여 생성되는
alias입니다. alias를 사용하지 않는 경우에는 `importlib`로 직접 로드할 수도
있습니다.

## 요구 사항

- transformers
- huggingface_hub
- pandas
- tqdm
- torch

이 모듈은 연구용 코드이며, 프로덕션 환경에서는 적절한 예외 처리와 로깅을
추가하여 사용하세요.
