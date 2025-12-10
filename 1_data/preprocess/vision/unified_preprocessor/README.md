# 통합 이미지 전처리 모듈

이 모듈은 하나의 입력 이미지에서 Swin Gray(밝은 배경 3채널)와 OCR(흰 배경 1채널) 이미지를
동시에 생성합니다. 탁본 검출, 텍스트 검출, 밝기/배경 보정, 크롭 등
다양한 처리를 포함하여 단일 호출로 최종 결과를 얻을 수 있습니다.

## 파일 구성

- **unified_preprocessor.py** – 통합 전처리 클래스(`UnifiedImagePreprocessor`)와 편의 함수(`preprocess_image_unified`)를 정의합니다.
- **preprocessor_config.json** – 전처리 파라미터와 파이프라인 설정을 담은 JSON 파일입니다.
- **\_\_init\_\_.py** – 패키지 초기화 모듈로 사용 예시를 제공합니다.

## 사용법

### 1. 편의 함수 사용

```python
from epitext_data.preprocess.vision.unified_preprocessor.unified_preprocessor import preprocess_image_unified

result = preprocess_image_unified(
    "input.jpg",
    "output_swin.jpg",
    "output_ocr.png",
    margin=10,
    use_rubbing=False,
)
if result["success"]:
    print("출력 경로:", result["swin"]["output_path"], result["ocr"]["output_path"])
else:
    print("오류:", result["message"])
```

### 2. 클래스 직접 사용

```python
from epitext_data.preprocess.vision.unified_preprocessor.unified_preprocessor import UnifiedImagePreprocessor

prep = UnifiedImagePreprocessor(config_path="preprocessor_config.json")
prep.preprocess_unified(
    "input.jpg",
    "swin.jpg",
    "ocr.png",
    margin=10,
    use_rubbing=False,
)
```

### 3. 설정 파일 편집

`preprocessor_config.json` 파일을 열어 `denoise_config`, `contrast_enhancement` 등
필요한 파라미터를 수정할 수 있습니다. 설명용 항목은 실제 설정에
영향을 주지 않습니다.

## 요구 사항

- opencv-python>=4.8.0
- numpy>=1.24.0

이 모듈은 연구용 코드이며, 서비스 환경에서는 적절한 예외 처리와
로깅을 추가하여 사용하세요.
