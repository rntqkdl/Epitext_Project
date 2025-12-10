# -*- coding: utf-8 -*-
'''구두점 복원 모델 패키지 초기화 모듈.

이 패키지는 한문 텍스트에서 누락된 구두점을 복원하기 위한 도구를
제공합니다. `main.py`의 `main()` 함수를 실행하면 CSV 데이터셋 전체의
구두점을 복원하고, `restore_punctuation_sliding` 함수를 사용하면 단일
문장의 구두점을 복원할 수 있습니다.

예제:

    from epitext_model.nlp.punctuation_restoration import config, main as punc_main
    from epitext_model.nlp.punctuation_restoration.main import restore_punctuation_sliding

    # 모델 다운로드 및 로드
    punc_main.download_model(config.MODEL_TAG, config.MODEL_CACHE_DIR)
    model_info = punc_main.load_model(config.MODEL_CACHE_DIR, device="cpu")

    # 단일 문장 처리
    cleaned = punc_main.remove_punctuation("예시 문장입니다")
    restored = restore_punctuation_sliding(cleaned, model_info)
    print(restored)

    # CSV 전체 처리
    punc_main.main()

위 예제에서 `epitext_model`은 `setup_import_aliases.py`를 실행하여
생성되는 alias 패키지입니다. alias를 사용하지 않는 경우에는
`importlib`을 통해 경로 기반으로 모듈을 로드할 수 있습니다.
'''
