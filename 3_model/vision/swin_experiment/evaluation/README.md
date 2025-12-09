
# Swin Transformer 평가 모듈 (Evaluation)



학습된 Swin Transformer 모델을 로드하여 한자 인식 성능을 검증하고 시각화하는 모듈입니다.



## 1. 주요 기능

* **모델 로드**: 체크포인트(.pth)에서 모델 가중치와 클래스 매핑 정보를 복원

* **추론 테스트**: 검증용 데이터셋(.npz)에서 무작위 샘플을 추출하여 Top-K 예측 수행

* **시각화**: 원본 이미지와 예측 결과를 비교하는 시각화 이미지 생성 (텍스트 겹침 해결)



## 2. 파일 구성

```text

evaluation/

├── __init__.py      # 패키지 초기화

├── config.py        # 설정 파일 (경로, 파라미터)

├── evaluate.py      # 메인 검증 스크립트

└── README.md        # 설명서

3. 사용 방법

설정 변경

config.py 파일에서 다음 항목을 사용자의 환경에 맞게 수정해야 합니다.



Python



# config.py 예시

CHECKPOINT_PATH = Path("C:/models/swin_checkpoint.pth")  # 모델 파일 경로

NPZ_PATHS = [Path("C:/data/val_data.npz")]               # 데이터 파일 경로

실행

터미널에서 아래 명령어로 실행합니다.



Bash



cd 3_model/vision/swin_experiment/evaluation

python evaluate.py

4. 출력 결과

콘솔 출력: Top-1 정확도 및 개별 샘플의 예측 결과(확률 포함)

이미지 저장: results/swin_top1_test.png (시각화 결과물)

5. 주의사항

한글 폰트: 시각화 시 한글 출력을 위해 시스템에 설치된 폰트(Malgun Gothic, NanumGothic 등)를 자동으로 사용합니다. 폰트가 없을 경우 깨질 수 있습니다.
