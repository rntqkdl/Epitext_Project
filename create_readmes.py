import os

# 기준 경로 설정
base_path = r"C:\hanja_data\Epitext_Project\5_docs\experiments"

# 실험 목록 정의 (실험명, 목적, 요약)
experiments = [
    {
        "path": "data/규장각_데이터_수집",
        "title": "규장각 데이터 수집",
        "purpose": "규장각 한국학 데이터베이스에서 고문서(탁본) 이미지 데이터를 크롤링하여 수집",
        "summary": "Selenium 자동화를 통해 페이지를 순차적으로 탐색하며 이미지 파일을 다운로드; 페이지가 자동으로 넘어가지 않는 문제를 해결하고 안정적인 대용량 다운로드 로직을 구현"
    },
    {
        "path": "nlp/sikuroberta/MLM_성능_비교",
        "title": "MLM 성능 비교",
        "purpose": "세 종류의 BERT 기반 언어 모델(SillokBERT, SikuRoBERTa, HUE)의 Masked Language Modeling 성능을 비교 평가",
        "summary": "자동으로 Top-1 및 Top-5 정확도를 측정하고, 일부 샘플에 대한 정성적(출력 비교) 평가를 수행하는 스크립트를 작성하여 모델 간 성능 차이를 분석"
    },
    {
        "path": "nlp/sikuroberta/판독문_전체_텍스트_중복_탁본_제외",
        "title": "판독문 전체 텍스트 중복 탁본 제외",
        "purpose": "조선왕조실록 판독문 전체 텍스트 코퍼스에서 중복된 탁본 데이터를 제외하고 텍스트 전처리 및 토크나이징",
        "summary": "토크나이저에 새로운 한자 토큰을 추가하여 Vocab을 확장하고, 중복이 제거된 텍스트로 SikuRoBERTa의 MLM 학습용 토큰화 데이터셋을 생성"
    },
    {
        "path": "nlp/sikuroberta/판독문_전체_텍스트_중복_탁본_포함",
        "title": "판독문 전체 텍스트 중복 탁본 포함",
        "purpose": "판독문 전체 텍스트 코퍼스를 중복된 탁본까지 모두 포함하여 전처리 및 토크나이징",
        "summary": "중복 데이터가 포함된 상태로 동일한 토큰화 과정을 진행하여 MLM 학습 데이터를 생성하고, 중복 제거 여부에 따른 모델 학습 영향도를 비교"
    },
    {
        "path": "nlp/gemini/ExaOne_성능_평가",
        "title": "ExaOne 성능 평가 코드",
        "purpose": "대규모 언어 모델 ExaOne을 활용하여 한문 금석문 번역 (음독, 고유명사 추출, 국역) 작업의 성능을 평가",
        "summary": "ExaOne 모델로 약 1000개의 한문 문장을 번역하고, 추출된 고유 명사에 기반한 BLEU 점수를 산출하여 번역 품질을 측정"
    },
    {
        "path": "nlp/gemini/Qwen_성능_평가",
        "title": "Qwen 성능 평가 코드",
        "purpose": "Alibaba의 Qwen-7B 언어 모델을 활용하여 한문 번역 작업의 성능을 평가",
        "summary": "ExaOne 실험과 동일한 프로세스로 Qwen 모델의 한문 번역을 실행하고 결과를 분석하여 두 모델의 번역 성능을 비교"
    },
    {
        "path": "image/opencv/OpenCV_이용한_이미지_전처리_1-6",
        "title": "OpenCV 이용한 이미지 전처리 1-6",
        "purpose": "OpenCV를 활용하여 탁본 이미지의 이진화 및 전처리 기법을 6차례에 걸쳐 시험 (이미지 가독성 향상 목적)",
        "summary": "일부 탁본 이미지에서 획과 배경이 제대로 분리되지 않는 문제를 확인하고, 대비 조절, 필터 적용 등 다양한 OpenCV 기반 방법을 시도하여 개선을 모색"
    },
    {
        "path": "image/briefnet/briefnet_을_이용한_이미지_전처리",
        "title": "briefnet을 이용한 이미지 전처리",
        "purpose": "최신 딥러닝 세그멘테이션 모델 BiRefNet을 활용하여 탁본 이미지의 배경을 제거하고 한자 부분만 분리",
        "summary": "탁본 이미지에 대해 경계선 기반 세그멘테이션을 수행하여 한자 영역을 추출하고, 생성된 마스크를 바탕으로 배경을 흰색으로 치환하여 글자만 강조"
    },
    {
        "path": "image/DBNet/DBNet_을_이용한_이미지_전처리",
        "title": "DBNet을 이용한 이미지 전처리",
        "purpose": "장면 텍스트 검출 모델 DBNet으로 고문서(탁본) 이미지의 한자 영역을 검출하고 배경을 제거",
        "summary": "DBNet으로 얻은 글자 영역 다각형을 마스크로 만들어 원본 이미지에서 해당 글자 부분만 남기고 배경은 흰색으로 변환 (저대비 이미지를 위해 CLAHE 대비 향상 기법 추가 적용)"
    },
    {
        "path": "image/U2Net/U²-Net_을_이용한_이미지_전처리",
        "title": "U²-Net을 이용한 이미지 전처리",
        "purpose": "사전 학습된 U²-Net 모델을 사용하여 탁본 이미지에서 한자 영역만 추출 (배경 제거)",
        "summary": "U²-Net으로 생성한 투명 마스크를 활용하여 글자 이외의 배경을 제거하고, 작은 잡영(pepper noise)을 제거한 뒤 글자만 남긴 흰 배경 이미지를 얻음"
    },
    {
        "path": "image/gan/Calligraphy_GAN",
        "title": "Calligraphy GAN",
        "purpose": "생성적 적대 신경망(GAN)을 활용하여 한자 서체 스타일 변환 또는 고문서 데이터 보강 가능성 탐색",
        "summary": "Calligraphy GAN 관련 외부 GitHub 프로젝트를 참고하여 한자 서체의 생성 및 복원을 실험적으로 검토 (원본 코드는 GitHub 링크를 통해 활용)"
    },
    {
        "path": "ocr/craft/CRAFT_을_이용한_한자_인식",
        "title": "CRAFT를 이용한 한자 인식",
        "purpose": "딥러닝 기반 문자 영역 검출 모델 CRAFT로 고문서 이미지에서 한자 영역을 탐지 및 분할",
        "summary": "CRAFT로 검출된 문자 영역 중 여러 글자가 붙은 덩어리는 거리 변환과 워터쉐드 알고리즘으로 개별 글자로 분리하여, 각 한자에 대한 bounding box를 추출"
    },
    {
        "path": "ocr/paddle/Paddle_OCR",
        "title": "Paddle OCR",
        "purpose": "PaddleOCR 라이브러리를 사용하여 탁본 이미지의 한자 인식을 수행",
        "summary": "사전 학습된 통합 OCR 엔진인 PaddleOCR(det+rec)을 활용해 한자 영역 검출 및 인식을 시도; 간편한 사용 대비 복잡한 한자 모양에 대한 인식 한계를 파악"
    },
    {
        "path": "ocr/paddle_ensemble/Paddle_OCR_과_고문서OCR_앙상블",
        "title": "Paddle OCR과 고문서OCR 앙상블",
        "purpose": "PaddleOCR와 HRNet 기반 자체 한자 OCR 결과를 결합(fusion)하여 인식 성능 향상을 도모",
        "summary": "세로쓰기 순서 보정 및 한 글자당 박스 분리 등을 적용한 뒤, PaddleOCR와 HRNet OCR의 예측 결과를 박스 단위로 가중치 앙상블하여 최종 한자 인식 정확도를 개선"
    },
    {
        "path": "ocr/aihub/AI_Hub_고문서_OCR_단독_실행_코드",
        "title": "AI Hub 고문서 OCR 단독 실행 코드",
        "purpose": "AI Hub에서 제공하는 한문 OCR 모델을 단독으로 적용하여 성능 및 특징 확인",
        "summary": "AI Hub의 공개 한문 OCR 모델로 입력 이미지를 처리하고 텍스트를 추출; PaddleOCR 등 다른 엔진과 비교하기 위해 단일 모델의 인식 결과를 평가"
    },
    {
        "path": "ocr/easyocr/Easy_OCR_한자_한_글자_인식",
        "title": "Easy OCR 한자 한 글자 인식",
        "purpose": "EasyOCR 오픈소스 OCR을 활용하여 이미지 내 개별 한자 글자의 인식 성능을 실험",
        "summary": "EasyOCR 엔진으로 탁본 이미지에서 한자를 한 글자씩 분리하여 인식하고 결과를 확인; 특정 한자에 대한 인식 정확도와 실용성을 평가"
    },
    {
        "path": "ocr/hrnet/Faster_RCNN_HRNet",
        "title": "Faster-RCNN + HRNet",
        "purpose": "딥러닝 객체 탐지 모델 Faster R-CNN에 HRNet 백본을 적용하여 한자 영역 검출 실험",
        "summary": "HRNet의 고해상도 특징 표현을 활용한 Faster R-CNN 모델로 탁본 이미지의 한자 영역을 검출하고, 그 정확도 및 효율성을 기존 방법과 비교"
    },
    {
        "path": "ocr/hrnet/Faster_RCNN_HRNet_Crop_augmentation",
        "title": "Faster R-CNN+HRNet Crop augmentation",
        "purpose": "Faster R-CNN+HRNet 기반 한자 검출에 이미지 크롭 증강 기법을 추가하여 성능 향상 시도",
        "summary": "모델 학습 및 추론 시 원본 이미지를 분할/증강하여 작은 글자도 검출할 수 있도록 개선; 데이터 증강을 통한 검출 성공률 변화를 관찰"
    },
    {
        "path": "ocr/fcos/FCOS",
        "title": "FCOS",
        "purpose": "앵커 없이 영역을 예측하는 1-stage 검출 모델 FCOS를 한자 문자 검출에 적용",
        "summary": "탁본 이미지 내 한자 위치를 FCOS 모델로 탐지하는 실험을 진행하고, anchor-free 접근법의 장단점 및 한자 검출 성능을 평가"
    },
    {
        "path": "ocr/yolo/YOLO",
        "title": "YOLO",
        "purpose": "실시간 객체 탐지 모델 YOLO를 활용하여 고문서 이미지의 한자 영역을 실시간 검출",
        "summary": "YOLOv5 기반 커스텀 모델로 한자 영역을 탐지하는 실험을 수행하고, 빠른 추론 속도 대비 한자 인식 정확도를 분석"
    },
    {
        "path": "ocr/kakren/Kakren_CHAT_OCR",
        "title": "Kakren(CHAT OCR)",
        "purpose": "대화형 OCR 시스템 'Kakren'을 사용한 한자 인식 가능성 탐색",
        "summary": "Kakren (ChatGPT 응용 OCR) 엔진으로 탁본 이미지의 한자를 인식해 보고, 최신 AI 기반 OCR의 성능과 활용 가능성을 검토"
    },
    {
        "path": "ocr/deepseek/DeepSeek_OCR",
        "title": "DeepSeek OCR",
        "purpose": "'DeepSeek OCR' 모델을 활용하여 탁본 이미지의 한자 인식 성능을 테스트",
        "summary": "DeepSeek OCR 엔진으로 한자를 추출하고 결과를 확인하여, 타 OCR 솔루션과의 인식률 및 정확도 비교에 활용"
    },
    {
        "path": "ocr/google/Google_OCR",
        "title": "Google OCR",
        "purpose": "Google Cloud Vision OCR API를 이용하여 탁본 이미지의 한자 인식을 시도",
        "summary": "Google OCR로 한자 이미지를 처리하여 텍스트를 추출하고, 상용 OCR 서비스의 한자 인식 품질과 한계점을 파악"
    }
]

# 각 실험별 폴더 생성 및 README.md 작성
created_files = []
for exp in experiments:
    dir_path = os.path.join(base_path, *exp["path"].split("/"))
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "README.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# {exp['title']}\n\n")
        f.write("## 목적\n")
        f.write(f"- {exp['purpose']}\n\n")
        f.write("## 시행착오 요약\n")
        f.write(f"- {exp['summary']}\n")
    created_files.append(file_path)

# 생성된 README.md 경로 출력
for path in created_files:
    print(path)
