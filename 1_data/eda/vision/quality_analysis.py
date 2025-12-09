"""
Image Quality Analysis & Filtering
======================================================================
목적: 이미지 품질 지표 분석, 이상치 탐지, 저품질 이미지 자동 제거
작성자: Epitext Project Team
======================================================================
"""

import sys
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 로컬 설정 임포트 시도
try:
    from config import Config
except ImportError:
    from .config import Config


# ======================================================================
# 데이터 로드 및 전처리
# ======================================================================
def load_and_clean_csv(csv_path):
    """
    CSV 로드 및 불필요한 컬럼 정리
    """
    print(f"[Info] Loading CSV: {csv_path}")
    
    if not csv_path.exists():
        print(f"[Error] File not found: {csv_path}")
        # 파일이 없으면 빈 데이터프레임 반환 대신 종료하지 않고 처리 유도 가능
        # 여기서는 안전하게 종료
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"[Info] Original shape: {df.shape}")

    drop_cols = ["stroke_width_consistency", "thin_weak_fraction"]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    if existing_drop_cols:
        df = df.drop(existing_drop_cols, axis=1)
        print(f"[Info] Dropped columns: {existing_drop_cols}")

    print(f"[Info] Cleaned shape: {df.shape}\n")
    return df


def basic_eda(df):
    """
    기본 기술 통계 및 결측치 확인
    """
    print("\n======================================================")
    print(" Basic Statistics")
    print("======================================================")
    print(df.describe())
    print("\n[Missing Values]")
    print(df.isnull().sum())
    print("======================================================\n")


# ======================================================================
# 시각화 함수
# ======================================================================
def plot_quality_distributions(df):
    """
    품질 지표별 히스토그램 시각화
    """
    for col in Config.QUALITY_COLS:
        if col not in df.columns:
            continue
            
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], bins=60, kde=True)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()


def plot_quality_correlation(df):
    """
    품질 지표 간 상관관계 히트맵
    """
    plt.figure(figsize=(12, 10))
    corr = df[Config.QUALITY_COLS].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Quality Metrics Correlation")
    plt.tight_layout()
    plt.show()


# ======================================================================
# 이상치 탐지 및 필터링
# ======================================================================
def iqr_bad_outlier_counts(df):
    """
    IQR 기반으로 품질이 나쁜 방향의 이상치 개수 계산
    """
    bad_outlier_counts = {}
    bounds = {}

    print("\n======================================================")
    print(" Outlier Analysis (IQR Method)")
    print("======================================================")

    for col in Config.QUALITY_COLS:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[col] = (lower, upper)

        if col in Config.BAD_HIGH_COLS:
            mask = df[col] > upper
        else:
            mask = df[col] < lower

        count = int(mask.sum())
        bad_outlier_counts[col] = count
        print(f" - {col}: {count} outliers (bounds={lower:.2f} ~ {upper:.2f})")

    print("======================================================\n")
    return bounds


def compute_bad_indicator_count(df, bounds):
    """
    각 이미지별로 나쁜 지표가 몇 개인지 계산
    """
    df = df.copy()
    bad_masks = []

    for col in Config.QUALITY_COLS:
        if col not in df.columns:
            continue
            
        lower, upper = bounds[col]
        if col in Config.BAD_HIGH_COLS:
            mask = df[col] > upper
        else:
            mask = df[col] < lower
        bad_masks.append(mask)

    if not bad_masks:
        df["bad_indicator_count"] = 0
        return df

    flag_matrix = np.column_stack(bad_masks)
    df["bad_indicator_count"] = flag_matrix.sum(axis=1)

    print("[Info] Bad Indicator Count Distribution:")
    print(df["bad_indicator_count"].value_counts().sort_index())
    print("")

    return df


def select_bad_images(df, threshold):
    """
    제거 대상 이미지 선택 (threshold 이상 나쁜 지표 보유 시)
    """
    drop_mask = df["bad_indicator_count"] >= threshold
    df_drop = df[drop_mask].copy()

    print(f"[Result] Images to remove (threshold={threshold}): {len(df_drop)}")
    print(f"[Result] Images to keep: {len(df) - len(df_drop)}\n")

    return df_drop


def move_bad_images(df_drop, src_dir, dst_dir):
    """
    저품질 이미지를 별도 폴더로 이동
    """
    if not df_drop.empty:
        dst_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    missing = 0

    print(f"[Info] Moving {len(df_drop)} images...")
    print(f" Source: {src_dir}")
    print(f" Dest:   {dst_dir}")

    for _, row in df_drop.iterrows():
        filename = row.get("filename")
        if not filename:
            continue
            
        src_path = src_dir / filename
        dst_path = dst_dir / filename

        if src_path.exists():
            try:
                shutil.move(str(src_path), str(dst_path))
                moved += 1
            except Exception as e:
                print(f"[Error] Failed to move {filename}: {e}")
        else:
            missing += 1

    print(f"[Done] Moved: {moved}, Missing: {missing}\n")


# ======================================================================
# 메인 실행 함수
# ======================================================================
def main():
    Config.print_config()
    
    # 1. 데이터 로드
    df = load_and_clean_csv(Config.CSV_PATH)

    # 2. 기본 EDA
    basic_eda(df)

    # 3. 시각화 (필요 시 주석 해제)
    # plot_quality_distributions(df)
    # plot_quality_correlation(df)

    # 4. 이상치 분석
    bounds = iqr_bad_outlier_counts(df)

    # 5. 나쁜 지표 개수 계산
    df = compute_bad_indicator_count(df, bounds)

    # 6. 제거 대상 선택
    df_drop = select_bad_images(df, threshold=Config.BAD_INDICATOR_THRESHOLD)

    # 7. 파일 이동
    move_bad_images(df_drop, Config.SRC_DIR, Config.DST_DIR)

    print("[Success] Quality analysis completed.")


if __name__ == "__main__":
    main()