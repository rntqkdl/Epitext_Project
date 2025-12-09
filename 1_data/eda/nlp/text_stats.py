"""
Text Data Statistical Analysis
======================================================================
목적: 텍스트 데이터의 기초 통계(길이, 빈도) 분석 및 Vocab 생성
작성자: Epitext Project Team
======================================================================
"""

import sys
import os
import re
from collections import Counter
import pandas as pd

# 로컬 설정 임포트
try:
    from config import Config
except ImportError:
    from .config import Config


# ======================================================================
# 데이터 로드 및 전처리
# ======================================================================
def load_data(file_path):
    """
    CSV 데이터 로드 및 결측치 처리
    """
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        # 파일이 없으면 빈 데이터프레임 반환 대신 예외 처리 유도
        return pd.DataFrame()
    
    print(f"[Info] Loading data: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return pd.DataFrame()

    # 분석할 컬럼 존재 여부 확인
    if Config.COLUMN_NAME not in df.columns:
        # 컬럼이 없을 경우, 첫 번째 컬럼을 텍스트 컬럼으로 가정하거나 경고
        print(f"[Warning] Column '{Config.COLUMN_NAME}' not found. Using first column as default.")
        target_col = df.columns[0]
    else:
        target_col = Config.COLUMN_NAME

    # 결측치 처리 및 문자열 변환
    df[target_col] = df[target_col].fillna("").astype(str)
    print(f"[Info] Loaded {len(df)} rows.")
    
    return df, target_col


# ======================================================================
# 통계 분석 함수
# ======================================================================
def analyze_length_stats(df, col_name):
    """
    문장 길이 통계 계산 (평균, 최대, 최소 등)
    """
    if df.empty:
        return df

    print("\n======================================================")
    print(" Sentence Length Statistics")
    print("======================================================")
    
    df["char_count"] = df[col_name].apply(len)
    stats = df["char_count"].describe()
    
    print(stats)
    print("======================================================\n")
    return df


def analyze_vocab(df, col_name, top_n=20):
    """
    글자 빈도 분석 (공백 및 특수문자 제외)
    """
    if df.empty:
        return Counter()

    print("[Info] Analyzing vocabulary frequency...")
    
    all_text = "".join(df[col_name].tolist())

    # 공백, 쉼표, 마침표 등 노이즈 제거 (한자/한글만 남기기 위해 정규식 조정 가능)
    # 현재는 기본적인 구두점만 제거
    cleaned_text = re.sub(r"[ ,。.\n\t]+", "", all_text)

    counter = Counter(cleaned_text)
    
    print(f"\n--- Top {top_n} Most Frequent Characters ---")
    most_common = counter.most_common(top_n)
    
    for rank, (char, count) in enumerate(most_common, 1):
        print(f" {rank}. {char}: {count}")
        
    print("-" * 30 + "\n")
    return counter


# ======================================================================
# 결과 저장
# ======================================================================
def save_vocab(counter, save_path):
    """
    Vocab 분석 결과를 CSV로 저장
    """
    if not counter:
        print("[Warning] No vocabulary to save.")
        return

    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    vocab_df = pd.DataFrame(counter.most_common(), columns=["char", "frequency"])
    vocab_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    print(f"[Success] Vocab saved to: {save_path}")
    print(f"Total Unique Characters: {len(vocab_df)}\n")


# ======================================================================
# 메인 실행 함수
# ======================================================================
def main():
    Config.print_config()
    
    # 1. 데이터 로드
    df, target_col = load_data(Config.INPUT_CSV)
    
    if df.empty:
        print("[Error] Dataframe is empty. Exiting.")
        return

    # 2. 데이터 미리보기
    print("--- Data Preview (Top 5) ---")
    print(df[[target_col]].head())
    print("-" * 30)

    # 3. 문장 길이 통계
    df = analyze_length_stats(df, target_col)

    # 4. Vocab 빈도 분석
    vocab_counter = analyze_vocab(df, target_col, Config.TOP_N_VOCAB)

    # 5. 결과 저장
    save_vocab(vocab_counter, Config.OUTPUT_VOCAB)


if __name__ == "__main__":
    main()