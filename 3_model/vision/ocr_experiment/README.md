# OCR 성능 평가 모듈 (OCR Evaluation Module)

이 모듈은 OCR 모델이 예측한 결과(JSON)와 실제 정답(TXT)을 비교하여 문자 단위 성능 지표를 계산합니다.

## 1. 주요 기능
* **CER (Character Error Rate) 계산**: 문자가 얼마나 틀렸는지 비율로 계산
* **Levenshtein Distance 분석**:
    * **Substitution (치환)**: 다른 글자로 잘못 인식한 경우
    * **Deletion (삭제)**: 글자를 인식하지 못한 경우
    * **Insertion (삽입)**: 없는 글자를 인식한 경우
* **유연한 JSON 파싱**: JSON 구조가 깊더라도 `text` 필드를 재귀적으로 모두 찾아 병합합니다.

## 2. 사용 방법

### (1) 명령줄 인터페이스 (CLI) 사용
터미널에서 직접 경로를 지정하여 실행하는 것을 권장합니다.

```bash
# 기본 사용법
python evaluate.py --gt "경로/정답파일.txt" --pred "경로/예측파일.json"

# 예시 (Windows)
python evaluate.py --gt "C:\data\gt_sample.txt" --pred "C:\results\output.json"
(2) 코드 내 설정 변경
config.py 파일의 DEFAULT_GT_PATH와 DEFAULT_PRED_PATH를 수정하여 기본값을 변경할 수 있습니다.

3. 입력 데이터 형식
정답 파일 (GT) - .txt
일반적인 텍스트 파일입니다. 줄바꿈이나 공백은 자동으로 제거되고 병합되어 평가됩니다.

Plaintext

(예시 content)
府尹嚴相公善政碑
公諱鼎耉
예측 파일 (Prediction) - .json
OCR 모델의 출력 결과입니다. 구조는 상관없으나 텍스트가 담긴 키 이름이 "text"여야 합니다.

JSON

{
    "images": [
        {
            "fields": [
                {"text": "府尹嚴相公善政碑", "confidence": 0.99},
                {"text": "公諱鼎耉", "confidence": 0.85}
            ]
        }
    ]
}
4. 참고용 예시 경로
아래 경로는 사용자 환경에 맞게 변경하여 사용하십시오.

GT 경로 예시: C:\ocr_test\gt.txt

Prediction 경로 예시: C:\ocr_test\ocr_results\KSM_NRICH_14746_img_4_damaged.json

5. 결과 해석
CER: 0에 가까울수록 좋습니다. (0.1 = 10% 에러)

Accuracy: 1에 가까울수록 좋습니다. (0.9 = 90% 정확도) '@ [System.IO.File]::WriteAllText("$PWD/3_model/vision/ocr_experiment/README.md", $readmeContent, [System.Text.Encoding]::UTF8)
$readmeContent = @'

# OCR 성능 평가 모듈 (OCR Evaluation Module)



이 모듈은 OCR 모델이 예측한 결과(JSON)와 실제 정답(TXT)을 비교하여 문자 단위 성능 지표를 계산합니다.



## 1. 주요 기능

* **CER (Character Error Rate) 계산**: 문자가 얼마나 틀렸는지 비율로 계산

* **Levenshtein Distance 분석**:

    * **Substitution (치환)**: 다른 글자로 잘못 인식한 경우

    * **Deletion (삭제)**: 글자를 인식하지 못한 경우

    * **Insertion (삽입)**: 없는 글자를 인식한 경우

* **유연한 JSON 파싱**: JSON 구조가 깊더라도 `text` 필드를 재귀적으로 모두 찾아 병합합니다.



## 2. 사용 방법



### (1) 명령줄 인터페이스 (CLI) 사용

터미널에서 직접 경로를 지정하여 실행하는 것을 권장합니다.



```bash

# 기본 사용법

python evaluate.py --gt "경로/정답파일.txt" --pred "경로/예측파일.json"



# 예시 (Windows)

python evaluate.py --gt "C:\data\gt_sample.txt" --pred "C:\results\output.json"

(2) 코드 내 설정 변경

config.py 파일의 DEFAULT_GT_PATH와 DEFAULT_PRED_PATH를 수정하여 기본값을 변경할 수 있습니다.

3. 입력 데이터 형식

정답 파일 (GT) - .txt

일반적인 텍스트 파일입니다. 줄바꿈이나 공백은 자동으로 제거되고 병합되어 평가됩니다.

Plaintext



(예시 content)

府尹嚴相公善政碑

公諱鼎耉

예측 파일 (Prediction) - .json

OCR 모델의 출력 결과입니다. 구조는 상관없으나 텍스트가 담긴 키 이름이 "text"여야 합니다.

JSON



{

    "images": [

        {

            "fields": [

                {"text": "府尹嚴相公善政碑", "confidence": 0.99},

                {"text": "公諱鼎耉", "confidence": 0.85}

            ]

        }

    ]

}

4. 참고용 예시 경로

아래 경로는 사용자 환경에 맞게 변경하여 사용하십시오.

GT 경로 예시: C:\ocr_test\gt.txt

Prediction 경로 예시: C:\ocr_test\ocr_results\KSM_NRICH_14746_img_4_damaged.json

5. 결과 해석

CER: 0에 가까울수록 좋습니다. (0.1 = 10% 에러)

Accuracy: 1에 가까울수록 좋습니다. (0.9 = 90% 정확도)
