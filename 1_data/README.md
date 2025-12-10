# 1_data - 데이터 파이프라인

데이터 수집, 전처리, EDA를 관리하는 모듈입니다.

## 구조

- `crawlers/`: 데이터 크롤링 코드
- `preprocess/`: 데이터 전처리 (NLP, Vision)
- `eda/`: 탐색적 데이터 분석
- `utils/`: 공통 유틸리티
- `sample_data/`: 샘플 데이터

## 실행 예시

모든 명령은 `Epitext_Project` 루트에서 실행합니다.

### 전체 파이프라인

```bash
python 1_data/main.py --step all
```

### 단계별 실행

```bash
# 크롤링만
python 1_data/main.py --step crawl

# 전처리만
python 1_data/main.py --step preprocess

# EDA만
python 1_data/main.py --step eda
```

### 상세 로그 출력

```bash
python 1_data/main.py --step all -v
```

## 프로젝트 구조

```
1_data/
  __init__.py          # 경로 설정
  config.py            # 실행 설정
  main.py              # 오케스트레이터
  README.md            # 이 파일
  crawlers/            # 크롤링 코드
  preprocess/          # 전처리 코드
  eda/                 # EDA 코드
  utils/               # 공통 유틸리티
  sample_data/         # 샘플 데이터
```

## 참고사항

- 각 단계의 실제 로직은 `config.py`의 플래그로 제어됩니다.
- 크롤링/전처리/EDA 모듈이 준비되면 main.py의 주석을 해제하면 됩니다.
