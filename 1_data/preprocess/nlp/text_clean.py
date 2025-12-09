"""
Text Preprocessing Module
======================================================================
목적: 한자 텍스트 노이즈 제거, 특수문자 정제, 길이 필터링 수행
작성자: Epitext Project Team
======================================================================
"""

import sys
import re
import pandas as pd
from tqdm import tqdm

# 로컬 설정 임포트 시도
try:
    from config import Config
except ImportError:
    from .config import Config


# ======================================================================
# 전처리 함수
# ======================================================================
def clean_text_base(line_text):
    """
    기본 텍스트 세정
    - 판독불가 기호 마스킹 (-> ▨)
    - 특수문자 및 노이즈 키워드 제거
    - 불필요한 공백 정리
    """
    if not isinstance(line_text, str):
        return ""

    try:
        # 1. 마스킹 토큰 처리
        text = re.sub(Config.SYMBOLS_TO_REPLACE, "▨", line_text)

        # 2. 특수문자 제거
        text = re.sub(Config.CHARS_TO_REMOVE, "", text)

        # 3. 노이즈 키워드 제거
        text = Config.NOISE_PATTERN.sub("", text)

        # 4. 공백 정리 (연속 공백 -> 단일 공백)
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = text.strip()

        return text
    except Exception:
        return ""


def flatten_text(text):
    """
    줄바꿈 제거 및 한 줄로 병합
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


# ======================================================================
# 메인 실행 함수
# ======================================================================
def main():
    Config.print_config()
    
    # 1. 데이터 로드
    if not Config.INPUT_CSV.exists():
        print(f"[Error] Input file not found: {Config.INPUT_CSV}")
        return

    print(f"[Info] Loading data: {Config.INPUT_CSV}")
    df = pd.read_csv(Config.INPUT_CSV)
    original_count = len(df)
    print(f"[Info] Total rows: {original_count}")

    # 2. 전처리 수행
    print("\n[Step 1] Cleaning & Flattening text...")
    tqdm.pandas(desc="Processing")
    
    # transcript 컬럼이 존재한다고 가정 (없으면 예외 처리 필요)
    target_col = "transcript"
    if target_col not in df.columns:
        print(f"[Error] Column '{target_col}' not found in CSV.")
        # 첫 번째 컬럼을 대체제로 사용
        target_col = df.columns[0]
        print(f"[Warning] Using '{target_col}' column instead.")

    df["preprocess"] = df[target_col].progress_apply(
        lambda x: flatten_text(clean_text_base(x))
    )

    # 3. 길이 필터링
    print(f"\n[Step 2] Filtering by length (Min: {Config.MIN_LENGTH})...")
    df_filtered = df[df["preprocess"].str.len() >= Config.MIN_LENGTH].copy()
    removed_count = original_count - len(df_filtered)

    print(f" -> Removed: {removed_count}")
    print(f" -> Remaining: {len(df_filtered)}")

    # 4. 결과 샘플 확인
    print("\n[Sample Data]")
    if not df_filtered.empty:
        sample = df_filtered.iloc[0]
        # doc_id 컬럼이 있는지 확인 후 출력
        doc_id = sample["doc_id"] if "doc_id" in sample else "Unknown_ID"
        print(f" ID: {doc_id}")
        print(f" Preprocessed: {sample['preprocess'][:100]}...")
    else:
        print(" No data remaining after filtering.")

    # 5. 저장
    df_filtered.to_csv(Config.OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[Success] Saved to: {Config.OUTPUT_CSV}")
    print(f" Columns: {list(df_filtered.columns)}")


if __name__ == "__main__":
    main()