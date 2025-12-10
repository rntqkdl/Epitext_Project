# Epitext Project – Research & Experiment Repository

본 저장소는 **한자 탁본 자동 복원 AI 시스템**을 연구·실험하기 위한 **연구 전용 코드 및 실험 관리 저장소**입니다.

서비스 운영을 위한 백엔드/프론트엔드 코드는 별도 저장소 [`rntqkdl/Epitext_Service`]에서 관리합니다.

---

## 📂 프로젝트 구조 (Project Structure)

이 저장소는 데이터 파이프라인부터 모델 학습, 실험 기록까지 체계적으로 구성되어 있습니다.

```text
Epitext_Project/
├── 1_data/                 # 데이터 파이프라인 (수집, 전처리, EDA)
│   ├── raw_data/           # (Git 제외) 원본 데이터 저장소 (.gitignore 적용)
│   ├── preprocess/         # Vision(EasyOCR) 및 NLP(Text Clean) 전처리 모듈
│   ├── crawlers/           # 데이터 수집 크롤러
│   └── eda/                # 데이터 분석 스크립트
│
├── 2_notebooks/            # 실험 및 프로토타이핑용 Jupyter Notebook
│
├── 3_model/                # 모델 학습 및 평가 파이프라인
│   ├── nlp/                # NLP 모델 (SikuRoBERTa, Gemini)
│   │   ├── sikuroberta/    # MLM 학습 및 평가
│   │   └── gemini_experiment/ # 번역 실험
│   ├── vision/             # Vision 모델 (Swin Transformer, OCR)
│   │   ├── swin_experiment/   # 이미지 복원 학습
│   │   └── ocr_experiment/    # OCR 성능 평가
│   └── saved_models/       # (Git 제외) 학습된 모델 가중치 저장소
│
├── 5_docs/                 # 연구 노트 및 과거 시행착오 기록 (Experiments Archive)
│
├── main.py                 # ✨ 통합 실행 컨트롤러 (Entry Point)
├── config.py               # (Optional) 프로젝트 전역 설정
├── requirements.txt        # 통합 의존성 패키지 목록
└── .env                    # (Git 제외) API Key 및 환경 변수 설정 파일
```

---

## 🚀 시작하기 (Getting Started)

### 1\. 환경 설정 및 의존성 설치

Python 3.9 이상 환경에서 실행하는 것을 권장합니다.

```bash
# 1. 저장소 클론
git clone [https://github.com/rntqkdl/Epitext_Project.git](https://github.com/rntqkdl/Epitext_Project.git)
cd Epitext_Project

# 2. 가상환경 생성 (Conda 권장)
conda create -n epitext python=3.10
conda activate epitext

# 3. 통합 의존성 설치 (필수)
# NLP, Vision, Crawling 관련 라이브러리가 모두 포함되어 있습니다.
pip install -r requirements.txt
```

### 2\. API 키 발급 및 설정 (상세 가이드)

본 프로젝트는 \*\*Gemini(번역)\*\*와 \*\*Google Cloud Vision(OCR)\*\*을 사용하므로 두 가지 키가 필요합니다.

#### **A. Google Gemini API Key 발급**

1.  [Google AI Studio](https://aistudio.google.com/app/apikey)에 접속하여 Google 계정으로 로그인합니다.
2.  좌측 상단의 **"Get API key"** 버튼을 클릭합니다.
3.  \*\*"Create API key in new project"\*\*를 클릭하여 키를 생성합니다.
4.  생성된 `AIza...`로 시작하는 키 문자열을 복사합니다.

#### **B. Google Cloud Vision API 키 (Service Account) 발급**

1.  [Google Cloud Console](https://console.cloud.google.com/)에 접속하여 새 프로젝트를 생성합니다.
2.  상단 검색창에 \*\*"Cloud Vision API"\*\*를 검색하고 **"사용(Enable)"** 버튼을 누릅니다.
3.  좌측 메뉴에서 \*\*[IAM 및 관리자] \> [서비스 계정]\*\*으로 이동합니다.
4.  \*\*"+ 서비스 계정 만들기"\*\*를 클릭하고 이름을 입력한 후 완료합니다.
5.  생성된 계정을 클릭하고 **[키(Keys)]** 탭으로 이동합니다.
6.  \*\*[키 추가] \> [새 키 만들기]\*\*를 클릭하고 유형을 **JSON**으로 선택하여 다운로드합니다.
7.  다운로드된 JSON 파일(예: `project-12345.json`)을 프로젝트 루트 폴더에 복사합니다.

#### **C. 환경 변수 파일(.env) 생성**

프로젝트 루트(`Epitext_Project/`)에 `.env` 파일을 생성하고 위에서 얻은 정보를 입력합니다.

**`.env` 파일 작성 예시:**

```env
# A. Gemini API Key (문자열 붙여넣기)
GOOGLE_API_KEY=AIzaSyD_Your_Gemini_Key_Here

# B. Google Cloud Vision JSON 파일 경로 (상대 경로)
GOOGLE_APPLICATION_CREDENTIALS=./your-project-key-12345.json
```

---

## 💻 실행 방법 (Usage)

프로젝트 루트의 \*\*`main.py`\*\*를 통해 데이터 전처리부터 모델 학습까지 **모든 파이프라인을 단일 명령어로 실행**할 수 있습니다.

### 1\. 데이터 파이프라인 (Data Pipeline)

데이터 전처리 및 분석을 수행합니다.

| 작업 단계       | 명령어                                          | 설명                                                                       |
| :-------------- | :---------------------------------------------- | :------------------------------------------------------------------------- |
| **전처리 통합** | `python main.py --phase data --step preprocess` | Vision(EasyOCR 필터링) 및 NLP(텍스트 정제) 전처리를 순차적으로 실행합니다. |
| **EDA**         | `python main.py --phase data --step eda`        | 데이터 통계 및 시각화 분석을 수행합니다.                                   |

### 2\. 모델 파이프라인 (Model Pipeline)

모델 학습 및 성능 평가를 수행합니다. `--task` 인자로 구체적인 작업을 지정합니다.

#### A. SikuRoBERTa (NLP - 한자 언어 모델)

```bash
# MLM 학습 (Fine-tuning)
python main.py --phase model --task sikuroberta_train

# 성능 평가 (Perplexity, Accuracy)
python main.py --phase model --task sikuroberta_eval
```

#### B. Swin Transformer (Vision - 이미지 복원)

```bash
# 학습 (Training)
python main.py --phase model --task swin_train

# 평가 (Evaluation)
python main.py --phase model --task swin_eval
```

#### C. Gemini (NLP - 번역 실험)

```bash
# 번역 및 정량 평가 (BLEU/BERTScore)
python main.py --phase model --task gemini_eval
```

---

## ⚙️ 설정 변경 (Configuration)

각 모듈의 하이퍼파라미터(Epoch, Batch Size, Learning Rate 등) 및 데이터 경로는 \*\*해당 모듈 내부의 `config.py`\*\*에서 관리합니다.

| 모듈                 | 설정 파일 경로                                 | 주요 설정 항목                            |
| :------------------- | :--------------------------------------------- | :---------------------------------------- |
| **SikuRoBERTa 학습** | `3_model/nlp/sikuroberta/train/config.py`      | Epochs, Batch Size, LR, Data Path         |
| **SikuRoBERTa 평가** | `3_model/nlp/sikuroberta/evaluation/config.py` | Model Path, Test Data Path                |
| **Swin 학습/평가**   | `3_model/vision/swin_experiment/config.py`     | Image Size, Augmentation, Checkpoint Path |
| **OCR 평가**         | `3_model/vision/ocr_experiment/config.py`      | GT/Pred Path                              |
| **Gemini 실험**      | `3_model/nlp/gemini_experiment/config.py`      | Model Version, Prompt Path                |

---

## 🧪 실험 기록 및 아카이브 (Archived Experiments)

과거의 시행착오 코드와 다양한 실험 기록은 **`5_docs/experiments/`** 폴더에 분류되어 보존되어 있습니다.

- **NLP Trials**: ExaOne, Qwen 번역 실험, 다양한 SikuRoBERTa 학습 조건 테스트
- **Vision Trials**: DeepSeek, Paddle, AIHub 등 다양한 OCR 모델 비교 실험, OpenCV 전처리 시행착오

---

## 📝 라이선스 및 출처

- **License**: MIT License
- **Data Source**: 서울대학교 규장각, 국사편찬위원회, 국립문화재연구소
- **Team**: 4조 복원왕 김탁본

<!-- end list -->

```

```
