"""
이미지 품질 지표 분석 및 이상치 제거
======================================================================
작성자: 4조 복원왕 김탁본
작성일: 2025-12-07
출처: 4주차 보고서
기능: 품질 지표 통계, 분포 시각화, 이상치 기반 필터링
======================================================================
"""

from pathlib import Path
import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# ======================================================================
# 경로 설정
# ======================================================================
BASE_DIR = Path(__file__).parent.parent.parent / "raw_data"
CSV_PATH = BASE_DIR / "image_quality_metrics.csv"
SRC_DIR = BASE_DIR / "filtered_takbon"
DST_DIR = BASE_DIR / "low_quality_removed"

# 분석 대상 품질 지표
QUALITY_COLS = [
    "illumination_variance",
    "global_contrast",
    "local_contrast",
    "blur_score",
    "smear_noise_ratio",
    "deterioration_mask_ratio",
    "bleed_through_likelihood",
]

# 나쁜 방향 정의
BAD_HIGH_COLS = [
    "illumination_variance",
    "blur_score",
    "smear_noise_ratio",
    "deterioration_mask_ratio",
    "bleed_through_likelihood",
]

BAD_LOW_COLS = [
    "global_contrast",
    "local_contrast",
]

BAD_INDICATOR_THRESHOLD = 2


# ======================================================================
# 유틸리티 함수
# ======================================================================
def load_and_clean_csv(csv_path):
    """CSV 로드 및 정리"""
    print(f"[INFO] CSV 로드: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"[INFO] 원본 데이터 크기: {df.shape}")

    drop_cols = ["stroke_width_consistency", "thin_weak_fraction"]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    if existing_drop_cols:
        df = df.drop(existing_drop_cols, axis=1)
        print(f"[INFO] 제거 컬럼: {existing_drop_cols}")

    print(f"[INFO] 정리 후 크기: {df.shape}\n")
    return df


def basic_eda(df):
    """기본 통계 출력"""
    print("\n--- 기술 통계 ---")
    print(df.describe())

    print("\n--- 데이터 정보 ---")
    print(df.info())

    print("\n--- 결측치 개수 ---")
    print(df.isnull().sum())
    print("\n")


def plot_quality_distributions(df):
    """품질 지표 분포 시각화"""
    for col in QUALITY_COLS:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], bins=60, kde=True)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()


def plot_quality_correlation(df):
    """상관관계 히트맵"""
    plt.figure(figsize=(12, 10))
    corr = df[QUALITY_COLS].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Quality Metrics Correlation")
    plt.tight_layout()
    plt.show()


def iqr_bad_outlier_counts(df):
    """IQR 기반 나쁜 방향 이상치 개수"""
    bad_outlier_counts = {}
    bounds = {}

    for col in QUALITY_COLS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[col] = (lower, upper)

        if col in BAD_HIGH_COLS:
            mask = df[col] > upper
        else:
            mask = df[col] < lower

        bad_outlier_counts[col] = int(mask.sum())

    print("\n--- IQR 기반 나쁜 이상치 개수 ---")
    for k, v in bad_outlier_counts.items():
        print(f"  - {k}: {v} (bounds={bounds[k]})")
    print("\n")

    return bad_outlier_counts, bounds


def compute_bad_indicator_count(df, bounds):
    """각 이미지별 나쁜 지표 개수 계산"""
    df = df.copy()
    bad_masks = {}

    for col in QUALITY_COLS:
        lower, upper = bounds[col]
        if col in BAD_HIGH_COLS:
            mask = df[col] > upper
        else:
            mask = df[col] < lower
        bad_masks[col] = mask

    flag_matrix = np.column_stack([bad_masks[col] for col in QUALITY_COLS])
    df["bad_indicator_count"] = flag_matrix.sum(axis=1)

    print("\n--- bad_indicator_count 분포 ---")
    print(df["bad_indicator_count"].value_counts().sort_index())
    print("\n")

    return df


def select_bad_images(df, threshold):
    """제거 대상 이미지 선택"""
    drop_mask = df["bad_indicator_count"] >= threshold
    df_drop = df[drop_mask].copy()

    print(f"\n[INFO] 제거 대상 (threshold={threshold}): {len(df_drop)}개")
    print(f"[INFO] 유지 대상: {len(df) - len(df_drop)}개\n")

    return df_drop


def move_bad_images(df_drop, src_dir, dst_dir):
    """저품질 이미지 이동"""
    dst_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0

    print("\n[INFO] 이미지 이동 시작")
    print(f"  - 소스: {src_dir}")
    print(f"  - 목적지: {dst_dir}")
    print(f"  - 대상: {len(df_drop)}개")

    for _, row in df_drop.iterrows():
        filename = row["filename"]
        src_path = src_dir / filename
        dst_path = dst_dir / filename

        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            moved += 1
        else:
            missing += 1

    print(f"\n[OK] 이동 완료: {moved}개")
    if missing > 0:
        print(f"[WARN] 누락: {missing}개\n")


# ======================================================================
# 메인 실행
# ======================================================================
def main():
    # CSV 로드
    df = load_and_clean_csv(CSV_PATH)

    # 기본 EDA
    basic_eda(df)

    # 분포 시각화
    plot_quality_distributions(df)

    # 상관관계
    plot_quality_correlation(df)

    # IQR 이상치 분석
    _, bounds = iqr_bad_outlier_counts(df)

    # 나쁜 지표 개수 계산
    df = compute_bad_indicator_count(df, bounds)

    # 제거 대상 선택
    df_drop = select_bad_images(df, threshold=BAD_INDICATOR_THRESHOLD)

    # 파일 이동
    move_bad_images(df_drop, SRC_DIR, DST_DIR)

    print("\n[COMPLETE] 품질 분석 완료\n")


if __name__ == "__main__":
    main()
