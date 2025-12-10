# -*- coding: utf-8 -*-
"""통합 이미지 전처리 실행 스크립트.

이 스크립트는 커맨드라인 인터페이스를 제공하여 단일 이미지에 대한
전처리를 수행합니다. 사용 예시는 다음과 같습니다.

    python -m epitext_data.preprocess.vision.unified_preprocessor.main \
        --input input.jpg \
        --output_swin output_swin.jpg \
        --output_ocr output_ocr.png \
        --config preprocessor_config.json

옵션:
    --input (-i): 입력 이미지 경로 (필수)
    --output_swin (-s): Swin 출력 이미지 경로 (필수)
    --output_ocr (-o): OCR 출력 이미지 경로 (필수)
    --config (-c): 설정 파일 경로 (선택)
    --margin (-m): 여백 값 (선택)
    --use_rubbing: 탁본 영역 검출 사용 여부 (선택)
"""

import argparse
from pathlib import Path
from .unified_preprocessor import preprocess_image_unified, UnifiedImagePreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="통합 이미지 전처리기")
    parser.add_argument('--input', '-i', required=True, help='입력 이미지 경로')
    parser.add_argument('--output_swin', '-s', required=True, help='Swin 출력 이미지 경로')
    parser.add_argument('--output_ocr', '-o', required=True, help='OCR 출력 이미지 경로')
    parser.add_argument('--config', '-c', default=None, help='설정 파일 경로')
    parser.add_argument('--margin', '-m', type=int, default=None, help='여백 값')
    parser.add_argument('--use_rubbing', action='store_true', help='탁본 영역 검출 사용 여부')
    parser.add_argument('--metadata', default=None, help='요약 JSON 저장 경로 (선택)')
    args = parser.parse_args()

    if args.config:
        prep = UnifiedImagePreprocessor(config_path=args.config)
        result = prep.preprocess_unified(
            args.input,
            args.output_swin,
            args.output_ocr,
            margin=args.margin,
            use_rubbing=args.use_rubbing,
            metadata_json_path=args.metadata,
        )
    else:
        result = preprocess_image_unified(
            args.input,
            args.output_swin,
            args.output_ocr,
            margin=args.margin,
            use_rubbing=args.use_rubbing,
            metadata_json_path=args.metadata,
        )
    print(result)


if __name__ == '__main__':
    main()
