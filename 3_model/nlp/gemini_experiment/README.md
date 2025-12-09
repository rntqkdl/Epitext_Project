
# Gemini NLP 실험 모듈 (Translation Experiment)



Gemini API를 활용하여 한문 금석문을 한국어로 번역하고, BLEU 및 BERTScore를 통해 번역 품질을 정량적으로 평가하는 모듈입니다.



## 1. 주요 기능

* **LLM 번역**: Gemini Pro/Flash 모델을 사용하여 Few-shot 프롬프팅 기반 번역 수행

* **BLEU 점수**: 기계 번역의 n-gram 일치도 기반 정량 평가

* **BERTScore**: KLUE/RoBERTa 모델을 활용한 문맥 유사도 기반 심층 평가



## 2. 파일 구성

```text

gemini_experiment/

├── __init__.py        # 패키지 초기화

├── config.py          # 설정 파일 (API 키, 경로, 모델)

├── prompts.py         # 시스템 프롬프트 및 예제 데이터

├── run_evaluation.py  # 메인 실행 스크립트

└── README.md          # 설명서

3. 사용 방법

설정 변경

config.py 파일에서 다음 항목을 수정할 수 있습니다.



INPUT_FILE: 번역할 원본 데이터셋 (CSV)

GEMINI_MODEL: 사용할 Gemini 모델 버전

TARGET_COUNT: 테스트할 샘플 개수

환경 변수 설정 (.env)

프로젝트 루트에 .env 파일을 생성하고 API 키를 입력해야 합니다.



Ini, TOML



GOOGLE_API_KEY=your_api_key_here

실행

터미널에서 아래 명령어로 실행합니다.



Bash



cd 3_model/nlp/gemini_experiment

python run_evaluation.py

4. 출력 결과

results/final_translation_bertscore.csv: 번역 결과 및 개별 점수

results/bertscore_model_summary.csv: 모델별 평균 성능 요약
