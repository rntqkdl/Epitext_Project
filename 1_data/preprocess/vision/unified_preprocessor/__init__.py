# -*- coding: utf-8 -*-
'''통합 이미지 전처리 패키지 초기화 모듈.

이 패키지는 하나의 입력 이미지로부터 Swin Gray(3채널)와 OCR(1채널) 이미지를
동시에 생성하는 기능을 제공합니다. `UnifiedImagePreprocessor` 클래스를 사용하거나
`preprocess_image_unified` 편의 함수를 이용할 수 있습니다.

예제:

    from epitext_data.preprocess.vision.unified_preprocessor.unified_preprocessor import (
        preprocess_image_unified, UnifiedImagePreprocessor
    )

    # 편의 함수 사용
    result = preprocess_image_unified(
        "input.jpg", "swin.jpg", "ocr.png", margin=10, use_rubbing=False
    )
    print(result)

    # 클래스 직접 사용
    prep = UnifiedImagePreprocessor(config_path="preprocessor_config.json")
    prep.preprocess_unified(
        "input.jpg", "swin.jpg", "ocr.png", margin=10, use_rubbing=False
    )

`epitext_data` alias 패키지는 `setup_import_aliases.py`를 실행하면 생성됩니다.
'''
