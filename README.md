# 🏛️ EpiText: 딥러닝 기반 훼손 탁본 복원 및 문맥 추론 AI 시스템 (AI Core Research & Experiment)

<div align="center">

[![Project Award](https://img.shields.io/badge/Award-대상_(정보통신기획평가원장상)-gold?style=for-the-badge&logo=trophy)](https://github.com/rntqkdl/Epitext_Project)
[![Academy](https://img.shields.io/badge/Academy-고려대학교_지능정보_SW아카데미_7기-004B9B?style=for-the-badge)](https://github.com/rntqkdl/Epitext_Project)
[![Model](https://img.shields.io/badge/Model-Swin_Transformer_V2-blueviolet?style=for-the-badge&logo=pytorch)](https://github.com/rntqkdl/Epitext_Project)
[![Top-1 Accuracy](https://img.shields.io/badge/Top--1_Acc-96.63%25-brightgreen?style=for-the-badge)](#-실측-벤치마크-performance-matrix)
[![Zero Middle Dot](https://img.shields.io/badge/Encoding-Zero_Middle_Dot_Safe-brightgreen?style=for-the-badge)](#)

<p align="center">
  <b>13,000종 한자 클래스의 극심한 롱테일 불균형과 심각한 물리적 노이즈를 극복한 탁본 이미지 복원 및 비전 AI 모델링 연구 아카이브</b>
</p>

</div>

---

## 🏆 주요 성과 및 학술 실적 (Key Achievements)

- **🥇 대상 (정보통신기획평가원장상)**: 고려대학교 지능정보 SW아카데미 최종 프로젝트 경진대회 **1위 대상 수상** (2025.12)
- **📄 학술지 논문 투고**: 한국통신학회(KICS) 및 한국정보기술학회(KIIT) 논문 투고 완료

---

## 🔗 관련 저장소 바로가기 (Repository Ecosystem)

본 저장소(`rntqkdl/Epitext_Project`)는 **EpiText 프로젝트의 핵심 AI 모델링, 손실 함수 수식 설계, 데이터 정제 및 성능 검증 실험**을 담당한 코어 리서치 아카이브입니다. 실서비스 운영을 위한 백엔드 및 프론트엔드 코드는 아래 팀 메인 저장소에서 확인하실 수 있습니다.

| 구분 | 저장소 링크 | 설명 및 담당 역할 |
|---|---|---|
| ⚙️ **메인 백엔드** | **[jincerity/Epitext_Back](https://github.com/jincerity/Epitext_Back)** | FastAPI 비동기 인퍼런스 서버, Docker 컨테이너라이징, Supabase DB 연동 |
| 💻 **메인 프론트엔드** | **[jincerity/Epitext_Front](https://github.com/jincerity/Epitext_Front)** | 탁본 이미지 업로드, Grad-CAM 히트맵 시각화, 한문 번역 인터랙티브 UI |
| 🔬 **리서치 및 모델링** | **[rntqkdl/Epitext_Project](https://github.com/rntqkdl/Epitext_Project)** | *(현재 저장소)* 13,000종 Swin V2 모델링, 커스텀 Loss 설계, 차등 학습률 실험 |

```
[ EpiText 통합 시스템 아키텍처 및 레포지토리 연결 맵 ]

  +-----------------------------------------------------------------------------------+
  | 🔬 AI Core Research & Experiment (This Repository)                                |
  |    👉 https://github.com/rntqkdl/Epitext_Project                                  |
  |    - 14,881개 원천 탁본 데이터 정제 및 13,000종 층화 추출 (Stratified Split)      |
  |    - 빈도수 제곱근 역수 기반 커스텀 손실 함수 (Custom Weighted Loss) 수식 설계     |
  |    - Swin Transformer V2 Backbone 및 2-Tier 차등 학습률 (Differential LR) 실험    |
  |    - Grad-CAM (XAI) 획(Stroke) 단위 활성화 시각화 및 모델 가중치(pth) 추출        |
  +-----------------------------------------+-----------------------------------------+
                                            │
                                            ▼ (학습된 최적 가중치 및 인퍼런스 파이프라인 배포)
  +-----------------------------------------------------------------------------------+
  | ⚙️ Production Backend API Server (Main)                                           |
  |    👉 https://github.com/jincerity/Epitext_Back                                   |
  |    - FastAPI 비동기 인퍼런스 서버 및 Docker 컨테이너라이징                        |
  |    - SikuRoBERTa 고전 문맥 추론 앙상블 파이프라인 서빙                             |
  |    - Supabase DB 연동 및 사용자 복원 히스토리 관리                                |
  +-----------------------------------------+-----------------------------------------+
                                            │
                                            ▼ (RESTful API 통신)
  +-----------------------------------------------------------------------------------+
  | 💻 Production Frontend Web Application (Main)                                     |
  |    👉 https://github.com/jincerity/Epitext_Front                                  |
  |    - 인터랙티브 탁본 이미지 업로드 및 실시간 복원 인터페이스                      |
  |    - Grad-CAM 판단 근거 히트맵 뷰어 및 한문 번역 텍스트 표출                       |
  +-----------------------------------------------------------------------------------+
```

---

## 🛠️ 기술 스택 (Tech Stack)

- **AI / Deep Learning**: Python, PyTorch, Swin Transformer V2, SikuRoBERTa, OpenCV, Hugging Face
- **Explainable AI (XAI)**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Data Engineering**: Pandas, NumPy, EasyOCR, Otsu Thresholding, Morphology Filter
- **Experiment & DevOps**: Git, GitHub, Conda, Jupyter Notebook, Docker

---

## 💡 프로젝트 핵심 가치 (Value Proposition)

- **문제 정의**: 오랜 세월 풍화와 마모로 훼손된 금석문/탁본 이미지는 획이 끊어지고 배경 노이즈가 심해 전문가도 판독에 수개월이 소요됨. 또한 13,000종의 방대한 클래스 중 상위 20개가 30%를 차지하고 대다수 희귀 한자는 1~2개에 불과한 극단적 롱테일 불균형 존재.
- **핵심 솔루션**:
  1. **수학적 가중치 손실 함수(Custom Weighted Loss)**를 설계하여 희귀 클래스 오분류에 동적 페널티 부여.
  2. **2-Tier 차등 학습률**을 적용하여 사전학습 시각 지식 보존과 13,000종 분류 헤드의 빠른 수렴 동시 달성.
  3. **Grad-CAM(설명 가능한 AI)**으로 복원 결과의 투명성과 학술적 신뢰성 확보.

---

## 📊 실측 벤치마크 (Performance Matrix)

| 모델 및 실험 구성 | Top-1 정확도 | Top-5 정확도 | 수렴 속도 (Convergence) | 희귀 클래스 탐지율 |
|---|:---:|:---:|:---:|:---:|
| **Baseline (ResNet50 + 일반 Cross-Entropy)** | 58.40% | 78.10% | Epoch 15 이후에도 진동 | 11.2% (심각한 편향) |
| **Vision Transformer (ViT-Base)** | 74.20% | 88.50% | Epoch 10 안정화 | 48.7% |
| **🏆 Swin V2 + Custom Weighted Loss (본 저장소)** | **96.63%** | **99.38%** | **Epoch 2 만에 조기 수렴** | **92.4% 달성** |

---

## 🔬 핵심 엔지니어링 기여 (Key Contributions)

### 1. 13,000종 불균형 해소를 위한 수학적 손실 함수 수식화
클래스 $c$의 샘플 수 $N_c$에 대해 빈도수 제곱근의 역수에 비례하는 가중치 행렬을 정의하고 배치 단위로 정규화하여 Cross-Entropy에 결합:

$$w_c = rac{1}{\sqrt{N_c} + \epsilon}$$

이를 통해 다수 클래스로의 과적합을 방지하고 희귀 한자에 대한 학습 그래디언트를 균형 있게 보정함.

### 2. 2-Tier 차등 학습률 (Differential Learning Rate) 전략
- **Backbone (Swin Transformer V2)**: 기학습된 시각 특징 유지를 위해 미세 학습률(`3e-5`) 적용.
- **Classification Head (13,000 Classes)**: 새로운 도메인 분류 레이어 적응을 위해 10배 높은 학습률(`3e-4`) 적용.
- **결과**: 학습 불안정성을 제거하고 Epoch 2에서 검증 정확도 41%p 급상승 달성.

### 3. 14,881개 원천 데이터 정제 파이프라인
- 원천 탁본 이미지 14,881개 중 다변량 EDA를 통해 이상치 106개를 제거하고 9,970개의 고품질 데이터셋으로 32.9% 정제.
- Grayscale 변환, Otsu 적응형 이진화, 모폴로지 연산을 통한 물리적 스캔 노이즈 제거 파이프라인 구축.

---

## 📂 프로젝트 구조 (Project Structure)

```text
Epitext_Project/
├── 1_data/                 # 데이터 파이프라인 (수집, 전처리, EDA)
│   ├── raw_data/           # (Git 제외) 원본 데이터 저장소 (.gitignore 적용)
│   ├── preprocess/         # Vision(EasyOCR) 및 NLP(Text Clean) 전처리 모듈
│   ├── crawlers/           # 데이터 수집 크롤러
│   └── eda/                # 데이터 분석 스크립트
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
├── 4_test_main/            # 단위 테스트 및 인퍼런스 검증
├── 5_docs/                 # 연구 노트 및 과거 시행착오 기록 (Experiments Archive)
│
├── main.py                 # ✨ 통합 실행 컨트롤러 (Entry Point)
├── requirements.txt        # 통합 의존성 패키지 목록
└── test.py                 # 빠른 모델 추론 테스트 스크립트
```

---

## 🚀 빠른 시작 (Getting Started)

### 1. 환경 설정 및 의존성 설치
Python **3.10 이상** 환경을 권장합니다.

```bash
# 1. 저장소 클론
git clone https://github.com/rntqkdl/Epitext_Project.git
cd Epitext_Project

# 2. 가상환경 생성 (Conda 권장)
conda create -n epitext python=3.10
conda activate epitext

# 3. 통합 의존성 패키지 설치
pip install -r requirements.txt
```

---

## 📁 학습 데이터 다운로드 (Google Drive)

대규모 학습 및 평가 데이터는 용량 관리 및 라이선스 이슈로 인해 Google Drive를 통해 제공됩니다.

- 🔗 **Google Drive 데이터 폴더**: [데이터셋 다운로드 링크](https://drive.google.com/drive/folders/1dqhfSy4_nnQTqXvZ3yqMpgbpR1r0nOkn?usp=drive_link)
- 다운로드한 데이터를 압축 해제 후 `1_data/raw_data/` 하위 경로에 배치해 주세요.
