"""
텍스트 데이터 통계 분석
======================================================================
작성자: 4조 복원왕 김탁본
작성일: 2025-12-07
출처: 4주차 보고서
기능: 문장 길이 통계, 글자 빈도 분석, Vocab 생성
======================================================================
"""

import pandas as pd
from collections import Counter
import re
import os
import sys


# ======================================================================
# 설정
# ======================================================================
INPUT_CSV = "doc_id_split_sentences.csv"
OUTPUT_VOCAB = "vocab.csv"


# ======================================================================
# 유틸리티 함수
# ======================================================================
def load_data(file_path):
    """CSV 데이터 로드 및 전처리"""
    if not os.path.exists(file_path):
        print(f"[Error] 파일을 찾을 수 없습니다: {file_path}")
        sys.exit(1)
    
    print(f"[INFO] 데이터 로딩: {file_path}")
    df = pd.read_csv(file_path)
    df["sentence"] = df["sentence"].fillna("").astype(str)
    print("[OK] 데이터 로드 완료\n")
    return df


def analyze_length_stats(df, col_name="sentence"):
    """문장 길이 통계 계산"""
    df["char_count"] = df[col_name].apply(len)

    print("--- 문장 길이 통계 ---")
    print(df["char_count"].describe())
    print("-" * 30 + "\n")
    
    return df


def analyze_vocab(df, col_name="sentence", top_n=20):
    """글자 빈도 분석 (공백/구두점 제외)"""
    print("[INFO] Vocab 빈도 분석 시작")
    all_text = "".join(df[col_name].tolist())

    # 공백, 쉼표, 마침표 제거
    cleaned_text = re.sub(r"[ ,。.]+", "", all_text)

    counter = Counter(cleaned_text)
    
    print(f"\n--- 가장 자주 등장하는 글자 Top {top_n} ---")
    most_common = counter.most_common(top_n)
    for char, count in most_common:
        print(f"'{char}': {count}회")
    print("-" * 30 + "\n")

    return counter


def save_vocab(counter, save_path):
    """Vocab 분석 결과 저장"""
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    vocab_df = pd.DataFrame(counter.most_common(), columns=["char", "frequency"])
    vocab_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    print(f"[OK] Vocab 저장 완료: {save_path}")
    print(f"총 Vocab 수: {len(vocab_df)}\n")


# ======================================================================
# 메인 실행
# ======================================================================
def main():
    # 데이터 로드
    df = load_data(INPUT_CSV)

    # 데이터 미리보기
    print("--- 데이터 미리보기 (상위 5행) ---")
    print(df.head())
    print("\n")

    # 문장 길이 통계
    df = analyze_length_stats(df)

    # Vocab 빈도 분석
    vocab_counter = analyze_vocab(df)

    # Vocab 저장
    save_vocab(vocab_counter, OUTPUT_VOCAB)


if __name__ == "__main__":
    main()
