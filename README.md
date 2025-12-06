# EPITEXT: 한자 탁본 자동 복원 시스템

## 프로젝트 개요
손상된 한자 탁본을 딥러닝으로 자동 복원하고 번역하는 종단간 AI 파이프라인

## 시스템 아키텍처
7단계 파이프라인: 전처리 → OCR → 구두점 복원 → 문맥 분석 → 시각 복원 → 번역

## 주요 모델
- **SikuRoBERTa**: 문맥 분석 (Top-5 90.3%)
- **Swin Transformer**: 시각 복원 (Top-1 68.4%)
- **Gemini 2.5**: 번역 (BLEU 0.59)

## 프로젝트 구조
Epitext_Project/
├── src/models/ # AI 모델
├── backend/ # FastAPI
├── frontend/ # React
├── 1_data/ # 데이터
├── 2_notebooks/ # 실험
├── 3_model/ # 학습
├── 4_system/ # 시스템
└── 5_docs/ # 문서

text

## 실행 방법
```bash
pip install -r requirements.txt
cd backend && python app.py
출처
Backend: https://github.com/jae2022/Epitext_Back

Frontend: https://github.com/jae2022/Epitext_Front

팀: 4조 복원왕 김탁본 (2025)
