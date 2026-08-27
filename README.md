# 🏛️ EpiText: 딥러닝 기반 훼손 탁본 복원 및 문맥 추론 AI 시스템 (AI Core Research & Experiment)

<div align="center">

[![Project Award](https://img.shields.io/badge/Award-대상_(정보통신기획평가원장상)-gold?style=for-the-badge&logo=trophy)](https://github.com/rntqkdl/Epitext_Project)
[![Academy](https://img.shields.io/badge/Academy-고려대학교_지능정보_SW아카데미_7기-004B9B?style=for-the-badge)](https://github.com/rntqkdl/Epitext_Project)
[![HF Model Vision](https://img.shields.io/badge/HuggingFace-ANSEONGMIN%2Fhanja--swinv2-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/ANSEONGMIN/hanja-swinv2)
[![HF Model NLP](https://img.shields.io/badge/HuggingFace-jhangyejin%2Fepitext--sikuroberta-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/jhangyejin/epitext-sikuroberta)
[![Top-1 Accuracy](https://img.shields.io/badge/Top--1_Acc-96.63%25-brightgreen?style=for-the-badge)](#-실측-벤치마크-performance-matrix)
[![Zero Middle Dot](https://img.shields.io/badge/Encoding-Zero_Middle_Dot_Safe-brightgreen?style=for-the-badge)](#)

<p align="center">
  <b>13,974종 한자 클래스의 극심한 롱테일 불균형과 심각한 물리적 노이즈를 극복한 탁본 이미지 복원 및 비전/NLP 멀티모달 AI 모델링 연구 아카이브</b>
</p>

</div>

---

## 🏆 주요 성과 및 학술 실적 (Key Achievements)

- **🥇 대상 (정보통신기획평가원장상)**: 고려대학교 지능정보 SW아카데미 최종 프로젝트 경진대회 **1위 대상 수상** (2025.12)
- **📄 학술지 논문 투고**: 한국통신학회(KICS) 및 한국정보기술학회(KIIT) 논문 투고 완료

---

## 🌐 EpiText 프로젝트 생태계 및 모델 허브 (Repository & Model Ecosystem)

본 저장소(`rntqkdl/Epitext_Project`)는 **EpiText 시스템의 13,974종 Swin V2 비전 모델링 및 SikuRoBERTa NLP 데이터 전처리/학습, 손실 함수 설계, 다변량 EDA 및 성능 검증 실험**을 총괄한 코어 리서치 아카이브입니다.

| 구분 | 저장소 / 모델 링크 | 기술 스택 및 담당 역할 |
|---|---|---|
| 👁️ **Vision AI Model** | **[HuggingFace: ANSEONGMIN/hanja-swinv2](https://huggingface.co/ANSEONGMIN/hanja-swinv2)** | 13,974종 고문서 한자 분류 모델 (Swin Transformer V2 Small) |
| 📖 **NLP Context Model** | **[HuggingFace: jhangyejin/epitext-sikuroberta](https://huggingface.co/jhangyejin/epitext-sikuroberta)** | 고전 한문 문맥 기반 구두점 복원 및 오인식 보정 (SikuRoBERTa MLM) |
| ⚙️ **메인 백엔드 API** | **[GitHub: jincerity/Epitext_Back](https://github.com/jincerity/Epitext_Back)** | FastAPI 비동기 인퍼런스 서버, Docker 컨테이너라이징, Supabase DB 연동 |
| 💻 **메인 프론트엔드 Web** | **[GitHub: jincerity/Epitext_Front](https://github.com/jincerity/Epitext_Front)** | 탁본 이미지 업로드, Grad-CAM 히트맵 시각화, 한문 번역 인터랙티브 UI |
| 🔬 **리서치 및 실험** | **[GitHub: rntqkdl/Epitext_Project](https://github.com/rntqkdl/Epitext_Project)** | *(현재 저장소)* 비전/NLP 모델 학습, 전처리 파이프라인, 성능 평가 |

```
[ EpiText 통합 시스템 아키텍처 및 파이프라인 흐름도 ]

  +-----------------------------------------------------------------------------------+
  | 🔬 AI Core Research & Experiment (This Repository)                                |
  |    👉 https://github.com/rntqkdl/Epitext_Project                                  |
  |    - 13,966개 원천 탁본 이미지 7대 품질 지표(IQR) 및 EasyOCR 선별                |
  |    - 8,759개 판독문 / 6,345개 번역문 기반 SikuRoBERTa MLM 학습 및 텍스트 전처리   |
  |    - 13,974종 한자 빈도수 제곱근 역수 커스텀 손실 함수 (Custom Weighted Loss) 설계|
  |    - Swin V2 Backbone (3e-5) / Head (3e-4) 2-Tier 차등 학습률 실험                |
  |    - Grad-CAM (XAI) 획(Stroke) 단위 활성화 시각화 및 모델 가중치(pth) 추출        |
  +-----------------------------------------+-----------------------------------------+
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
  +-------------------------------------+   +-------------------------------------+
  | 🤗 Vision Model Hub                 |   | 🤗 NLP Model Hub                    |
  |   ANSEONGMIN/hanja-swinv2           |   |   jhangyejin/epitext-sikuroberta    |
  |   - 13,974 Classes (Top-1 96.63%)   |   |   - 문맥 기반 구두점 복원 및 한자 보정 |
  +------------------┬------------------+   +------------------┬------------------+
                     │                                         │
                     └──────────────────────┬──────────────────┘
                                            ▼ (앙상블 인퍼런스 서빙)
  +-----------------------------------------------------------------------------------+
  | ⚙️ Production Backend API Server                                                 |
  |    👉 https://github.com/jincerity/Epitext_Back (FastAPI / Docker / Supabase)      |
  +-----------------------------------------+-----------------------------------------+
                                            │
                                            ▼ (RESTful API)
  +-----------------------------------------------------------------------------------+
  | 💻 Production Frontend Web Application                                            |
  |    👉 https://github.com/jincerity/Epitext_Front (React Interactive UI)            |
  +-----------------------------------------------------------------------------------+
```

---

## 💡 프로젝트 핵심 가치 (Value Proposition)

- **문제 정의**: 오랜 세월 풍화와 마모로 훼손된 금석문/탁본 이미지는 획이 끊어지고 배경 노이즈가 심해 전문가도 판독에 수개월이 소요됨. 또한 13,974종의 방대한 클래스 중 상위 빈출 한자가 다수를 차지하고 대다수 희귀 한자는 1~2개에 불과한 극단적 롱테일 불균형 존재.
- **핵심 솔루션**:
  1. **수학적 가중치 손실 함수(Custom Weighted Loss)**: 희귀 클래스 오분류에 동적 페널티를 부여하여 롱테일 편향 극복.
  2. **2-Tier 차등 학습률**: Swin V2 Backbone(`3e-5`)과 Head(`3e-4`) 차등화로 사전학습 지식 보존 및 조기 수렴 달성.
  3. **멀티모달 앙상블 (Swin V2 + SikuRoBERTa)**: 비전 모델의 Top-5 예측 결과를 고전문헌 특화 언어 모델로 문맥 보정(Post-Correction).
  4. **Grad-CAM(설명 가능한 AI)**: 획(Stroke) 단위 판단 근거 히트맵 시각화로 인문학 연구 신뢰성 확보.

---

## 📊 실측 벤치마크 (Performance Matrix)

| 모델 및 실험 구성 | Top-1 정확도 | Top-5 정확도 | 수렴 속도 (Convergence) | 희귀 클래스 탐지율 |
|---|:---:|:---:|:---:|:---:|
| **Baseline (ResNet50 + 일반 Cross-Entropy)** | 58.40% | 78.10% | Epoch 15 이후에도 진동 | 11.2% (심각한 편향) |
| **Vision Transformer (ViT-Base)** | 74.20% | 88.50% | Epoch 10 안정화 | 48.7% |
| **🏆 Swin V2 + Custom Weighted Loss (본 저장소)** | **96.63%** | **99.38%** | **Epoch 2 만에 조기 수렴** | **92.4% 달성** |

---

## 🔬 핵심 엔지니어링 기여 (Key Contributions)

### 1. 13,974종 불균형 해소를 위한 수학적 손실 함수 수식화
클래스 $c$의 샘플 수 $N_c$에 대해 빈도수 제곱근의 역수에 비례하는 가중치 행렬을 정의하고 배치 단위로 정규화하여 Cross-Entropy에 결합:

$$w_c = rac{1}{\sqrt{N_c} + \epsilon}$$

이를 통해 다수 클래스로의 과적합을 방지하고 희귀 한자에 대한 학습 그래디언트를 균형 있게 보정함.

### 2. 2-Tier 차등 학습률 (Differential Learning Rate) 전략
- **Backbone (Swin Transformer V2)**: 기학습된 시각 특징 유지를 위해 미세 학습률(`3e-5`) 적용.
- **Classification Head (13,974 Classes)**: 새로운 도메인 분류 레이어 적응을 위해 10배 높은 학습률(`3e-4`) 적용.
- **결과**: Effective Batch Size 576(192 $	imes$ 3 Acc) 및 AMP 환경에서 Epoch 2 만에 검증 정확도 41%p 급상승 달성.

### 3. 다차원 데이터 수집 및 7대 품질 지표 전처리 파이프라인
- **데이터 통합**: 규장각, 국사편찬위원회, 금석문 조사보고서(2018~2023), 지식이음 등 총 **13,966개 탁본 이미지**, **8,759개 판독문**, **6,345개 번역문** 통합.
- **품질 지표(EDA)**: 조명 불균형, 명암비, 블러, 노이즈, 마스크 비율 등 7가지 품질 지표에 대한 IQR 기반 이상치 필터링 수행.
- **텍스트 정제**: 6대 표준 시대(고려, 조선, 삼국/남북국, 근현대, 선사/고대, 시대미상) 버킷 정규화 및 한글/만주문자/일본어 한자 노이즈 제거.

---

## 📂 프로젝트 구조 (Project Structure)

```text
Epitext_Project/
├── 1_data/                 # 데이터 파이프라인 (수집, 전처리, EDA)
│   ├── raw_data/           # (Git 제외) 원본 데이터 저장소 (.gitignore 적용)
│   ├── preprocess/         # Vision(EasyOCR, 통합 전처리) 및 NLP 텍스트 정제
│   ├── crawlers/           # 규장각, 국사편찬위원회, PDF 크롤러
│   └── eda/                # 7대 이미지 품질 지표 분석 및 NLP EDA
│
├── 3_model/                # 모델 학습 및 평가 파이프라인
│   ├── nlp/                # SikuRoBERTa MLM 학습, 구두점 복원, Gemini 번역
│   ├── vision/             # Swin Transformer V2 13,974종 학습 및 OCR 평가
│   └── saved_models/       # 모델 체크포인트 및 인퍼런스 파이프라인
│
├── 4_test_main/            # 단위 테스트 및 인퍼런스 검증
├── 5_docs/                 # 연구 노트, 실험 보고서 아카이브
│
├── main.py                 # ✨ 통합 실행 컨트롤러 (Entry Point)
├── requirements.txt        # 통합 의존성 패키지 목록
└── test.py                 # 모델 추론 테스트 스크립트
```

---

## 🚀 빠른 시작 (Getting Started)

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
