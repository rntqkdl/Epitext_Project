"""
Gemini Translation & Evaluation Pipeline
======================================================================
목적: Gemini API를 활용한 한문 금석문 번역 수행 및 BLEU/BERTScore 자동 평가
작성자: Epitext Project Team
날짜: 2025-12-09
======================================================================
"""

import sys
import time
import re
import warnings
from pathlib import Path

# 프로젝트 루트 경로 설정
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import sacrebleu
from tqdm import tqdm
from kiwipiepy import Kiwi
from bert_score import score
from google import genai
from google.genai import types

# 로컬 설정 파일 import
try:
    from config import Config
    from prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
except ImportError:
    # 실행 위치에 따른 상대 경로 처리
    from .config import Config
    from .prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES

# 경고 메시지 무시 설정
warnings.filterwarnings("ignore")


# ======================================================================
# 유틸리티 클래스
# ======================================================================
class TextUtils:
    """텍스트 전처리 및 파싱을 위한 유틸리티"""
    
    def __init__(self):
        self.kiwi = Kiwi()

    def get_full_prompt(self, input_text: str) -> str:
        """Few-shot 예제와 입력 텍스트를 결합하여 프롬프트 생성"""
        return f"{FEW_SHOT_EXAMPLES}\n<new_translation>\n<input>{input_text}</input>\n<output>"

    def extract_nouns(self, text: str) -> str:
        """BLEU 평가를 위해 Kiwi 형태소 분석기로 명사만 추출"""
        if not isinstance(text, str): 
            return ""
        
        # 괄호 및 특수문자 제거
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"[^가-힣\s]", "", text)
        
        try:
            tokens = self.kiwi.tokenize(text)
            nouns = [t.form for t in tokens if t.tag in {"NNG", "NNP", "NR", "NP", "NNB"}]
            return " ".join(nouns)
        except Exception:
            return ""

    @staticmethod
    def parse_gemini_output(full_output: str) -> str:
        """모델 출력에서 최종 번역 부분만 파싱"""
        try:
            if "[최종 번역]:" in full_output:
                return full_output.split("[최종 번역]:")[1].strip().split("\n")[0]
            elif "[최종 번역]" in full_output:
                return full_output.split("[최종 번역]")[1].strip().split("\n")[0]
            return full_output
        except Exception:
            return full_output

    @staticmethod
    def remove_hanja_for_bertscore(text: str) -> str:
        """BERTScore 평가용 텍스트 정제 (한자 병기 제거)"""
        if pd.isna(text): 
            return ""
        # (漢字) 형태 제거
        cleaned = re.sub(r"\([^\)]*[\u4e00-\u9fff][^\)]*\)", "", str(text))
        return re.sub(r"\s+", " ", cleaned).strip()


# ======================================================================
# 단계별 실행 함수
# ======================================================================
def run_translation_step(config, utils):
    """Step 1: Gemini API 번역 및 BLEU 평가"""
    print("\n" + "=" * 70)
    print("[Step 1] Gemini 번역 및 BLEU 평가 시작")
    print("=" * 70)

    # API 키 확인
    if not config.API_KEY:
        print("[ERROR] API Key가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return None

    # 입력 파일 확인
    input_path = config.DATA_DIR / config.INPUT_FILE
    if not input_path.exists():
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다: {input_path}")
        return None

    # 클라이언트 초기화
    try:
        client = genai.Client(api_key=config.API_KEY)
    except Exception as e:
        print(f"[ERROR] 클라이언트 초기화 실패: {e}")
        return None

    # 데이터 로드 및 샘플링
    df = pd.read_csv(input_path)
    sample_size = min(config.TARGET_COUNT, len(df))
    target_df = df.sample(n=sample_size, random_state=42).copy()

    print(f"입력 파일: {input_path}")
    print(f"처리 대상: {sample_size}건")

    # 결과 디렉토리 생성
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results_buffer = []

    # 번역 루프
    print("\n--- 번역 진행 중 ---")
    for idx, row in tqdm(target_df.iterrows(), total=sample_size, desc="Processing"):
        src = row.get("pun_transcription", "")
        ref = row.get("translation", "")

        if pd.isna(src) or str(src).strip() == "": 
            continue

        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=utils.get_full_prompt(src),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=1000
                )
            )
            full_out = resp.text.strip()
            # RPM 제한 준수 (2초 대기)
            time.sleep(2) 
        except Exception as e:
            print(f"[WARNING] API Error at index {idx}: {e}")
            full_out = ""

        # 결과 파싱
        hyp_clean = utils.parse_gemini_output(full_out)
        
        # BLEU 점수 계산
        ref_nouns = utils.extract_nouns(ref)
        hyp_nouns = utils.extract_nouns(hyp_clean)
        
        bleu_score = 0.0
        if ref_nouns and hyp_nouns:
            bleu_score = sacrebleu.sentence_bleu(hyp_nouns, [ref_nouns], tokenize="char").score

        results_buffer.append({
            "doc_id": row.get("doc_id", idx),
            "src_hanja": src,
            "ref_korean": ref,
            "hyp_clean": hyp_clean,
            "bleu_score": bleu_score
        })

    # 결과 저장
    result_df = pd.DataFrame(results_buffer)
    output_path = config.RESULT_DIR / config.OUTPUT_FILE
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n[INFO] 1차 번역 결과 저장 완료: {output_path}")
    return result_df


def run_bertscore_step(df, config, utils):
    """Step 2: BERTScore 심층 평가"""
    print("\n" + "=" * 70)
    print("[Step 2] BERTScore 심층 평가 시작")
    print("=" * 70)

    # 데이터프레임 로드 확인
    if df is None or df.empty:
        output_path = config.RESULT_DIR / config.OUTPUT_FILE
        if output_path.exists():
            print(f"기존 결과 파일 로드: {output_path}")
            df = pd.read_csv(output_path)
        else:
            print("[ERROR] 평가할 데이터가 없습니다.")
            return

    # 전처리 (한자 제거)
    print("\n--- 텍스트 정제 (한자 병기 제거) ---")
    df["ref_filtered"] = df["ref_korean"].apply(utils.remove_hanja_for_bertscore)
    df["hyp_filtered"] = df["hyp_clean"].apply(utils.remove_hanja_for_bertscore)
    
    valid_mask = (df["ref_filtered"].str.len() > 0) & (df["hyp_filtered"].str.len() > 0)
    valid_df = df[valid_mask]
    
    refs = valid_df["ref_filtered"].tolist()
    hyps = valid_df["hyp_filtered"].tolist()
    
    print(f"유효 평가 데이터: {len(refs)}건")
    
    model_summary = []
    
    # 모델별 평가 루프
    for model_cfg in config.BERTSCORE_MODELS:
        model_name = model_cfg["name"]
        print(f"\n>>> 모델 평가 수행: {model_name}")
        
        try:
            P, R, F1 = score(
                hyps, refs,
                model_type=model_cfg["model_type"],
                num_layers=model_cfg["num_layers"],
                lang=model_cfg["lang"],
                batch_size=32,
                verbose=True
            )
            
            mean_f1 = F1.mean().item()
            model_summary.append({
                "model": model_name,
                "f1_mean": mean_f1,
                "precision_mean": P.mean().item(),
                "recall_mean": R.mean().item()
            })
            
            # 개별 점수 기록
            col_id = model_name.replace(" ", "_").lower() + "_f1"
            df.loc[valid_df.index, col_id] = F1.numpy()
            
            print(f"   -> Average F1 Score: {mean_f1:.4f}")
            
        except Exception as e:
            print(f"[ERROR] 모델 평가 실패 ({model_name}): {e}")

    # 최종 결과 저장
    final_path = config.RESULT_DIR / config.OUTPUT_FILE
    df.to_csv(final_path, index=False, encoding="utf-8-sig")
    
    summary_path = config.RESULT_DIR / config.SUMMARY_FILE
    pd.DataFrame(model_summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 70)
    print("평가 완료")
    print("=" * 70)
    print(f"상세 결과: {final_path}")
    print(f"요약 결과: {summary_path}")


# ======================================================================
# 메인 함수
# ======================================================================
def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("Gemini 번역 및 평가 파이프라인")
    print("=" * 70)
    
    # 설정 출력
    Config.print_config()
    
    # 유틸리티 초기화
    utils = TextUtils()
    
    # Step 1: 번역
    result_df = run_translation_step(Config, utils)
    
    # Step 2: 평가
    if result_df is not None:
        run_bertscore_step(result_df, Config, utils)
    else:
        print("\n[ERROR] 번역 단계 실패로 평가를 중단합니다.")

    print("\n" + "=" * 70)
    print("모든 작업 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()