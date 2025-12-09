
# Swin Transformer V2 학습 모듈

이 모듈은 한자 탁본 이미지를 인식하기 위해 Swin Transformer V2 모델을 학습시키는 파이프라인입니다.

## 1. 사전 준비 사항

### 하드웨어 권장 사양
* **GPU**: NVIDIA A100 (40GB+) 또는 동급 이상 권장
* **RAM**: 64GB 이상 (High-RAM 환경 필수)
* **저장 공간**: 학습 데이터 크기에 따라 다름

### 데이터 준비 (.npz 포맷)
이 코드는 학습 속도 최적화를 위해 이미지를 `.npz` 포맷으로 샤딩(Sharding)하여 사용합니다.
데이터 폴더(`data/`) 내에 아래 구조로 파일을 준비해야 합니다.

data/
├── train_shards/
│   ├── train_part001.npz
│   ├── train_part002.npz
│   └── ...
├── val_shards/
│   ├── val_part001.npz
│   └── ...
└── char_mapping.json  (클래스 매핑 정보)

## 2. 사용 방법
### (1) 설정 변경
`config.py` 파일을 열어 데이터 경로를 수정하십시오.
기본적으로 프로젝트 내부의 `1_data/processed/swin_data`를 바라봅니다.

```python
# config.py
DATA_BASE_DIR = PROJECT_ROOT / "my_custom_data_path"

(2) 학습 실행
터미널에서 아래 명령어로 학습을 시작합니다.
Bash
cd 3_model/vision/swin_experiment
python train.py

3. 주요 기능
메모리 효율적 로딩: 대용량 NPZ 데이터를 메모리에 로드하여 I/O 병목을 제거합니다.
불균형 데이터 대응: 클래스별 빈도 역수를 가중치로 사용하는 Loss Function을 적용했습니다.
자동 체크포인트: Best F1 Score 모델을 자동으로 저장하며, Early Stopping을 지원합니다.
모니터링: TensorBoard를 통해 실시간 학습 현황을 확인할 수 있습니다.
