# -*- coding: utf-8 -*-
"""
OCR Character Error Rate (CER) Evaluation Script
======================================================================
프로젝트: EPITEXT AI - 한자 탁본 복원
모듈: OCR 성능 평가 (CER 및 Levenshtein Distance)
작성자: Epitext Project Team
작성일: 2025-12-09
======================================================================
기능:
1. CER (Character Error Rate) 계산
2. Levenshtein Distance 기반 상세 분석 (치환/삭제/삽입)
3. JSON 예측 결과 파싱 (재귀적 탐색)
======================================================================
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np

# 로컬 설정 모듈 임포트 시도
try:
    from config import Config
except ImportError:
    from .config import Config


# ======================================================================
# 1. Levenshtein Distance 및 CER 계산 로직
# ======================================================================
def calculate_cer(gt: str, pred: str) -> Tuple[float, int, int, int]:
    """
    CER 및 편집 거리 세부 지표(S/D/I)를 계산합니다.

    Args:
        gt (str): 정답 문자열 (Ground Truth)
        pred (str): 예측 문자열 (Prediction)

    Returns:
        Tuple[float, int, int, int]:
            - cer_value: CER 값 (0.0 ~ 1.0 이상)
            - substitutions (S): 치환 횟수
            - deletions (D): 삭제 횟수
            - insertions (I): 삽입 횟수
    """
    if len(gt) == 0:
        # GT가 비어있을 경우, pred 길이만큼 삽입된 것으로 간주할 수도 있으나,
        # 일반적으로 CER 계산에서는 분모가 0이 되므로 예외 처리
        return 0.0, 0, 0, len(pred)

    m, n = len(gt), len(pred)

    # DP 테이블 초기화
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)

    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    # 거리 계산
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,        # 삭제 (Deletion)
                dp[i, j - 1] + 1,        # 삽입 (Insertion)
                dp[i - 1, j - 1] + cost  # 치환 (Substitution) 또는 일치
            )

    dist = int(dp[m, n])

    # 역추적 (Backtracking)하여 S/D/I 카운트
    i, j = m, n
    s_count = d_count = i_count = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if gt[i - 1] == pred[j - 1] and dp[i, j] == dp[i - 1, j - 1]:
                # 일치
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i - 1, j - 1] + 1:
                # 치환
                s_count += 1
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i - 1, j] + 1:
                # 삭제
                d_count += 1
                i -= 1
            elif dp[i, j] == dp[i, j - 1] + 1:
                # 삽입
                i_count += 1
                j -= 1
            else:
                i -= 1
                j -= 1
        elif i > 0:
            # 남은 GT 삭제
            d_count += 1
            i -= 1
        else:  # j > 0
            # 남은 Pred 삽입
            i_count += 1
            j -= 1

    cer_value = dist / len(gt)
    return cer_value, s_count, d_count, i_count


# ======================================================================
# 2. 파일 처리 유틸리티
# ======================================================================
def load_gt_text(gt_path: Path) -> str:
    """
    GT 텍스트 파일을 로드하여 전처리합니다.
    공백과 개행을 제거하고 하나의 문자열로 병합합니다.
    """
    try:
        lines = []
        with gt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        return "".join(lines)
    except Exception as e:
        print(f"[ERROR] GT 파일 로드 실패: {e}")
        return ""


def _extract_text_recursive(obj: Any, collector: List[str]) -> None:
    """JSON 객체 내부를 재귀적으로 순회하며 text 필드를 추출합니다."""
    if isinstance(obj, dict):
        text_val = obj.get("text")
        if isinstance(text_val, str):
            collector.append(text_val)
        
        for v in obj.values():
            _extract_text_recursive(v, collector)

    elif isinstance(obj, list):
        for v in obj:
            _extract_text_recursive(v, collector)


def load_pred_text(pred_path: Path) -> str:
    """
    예측 결과 JSON 파일을 로드하여 텍스트를 추출합니다.
    JSON 구조에 상관없이 text 키를 가진 모든 값을 수집합니다.
    """
    try:
        with pred_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        _extract_text_recursive(data, texts)

        if not texts:
            # 텍스트 필드가 없는 경우 전체 덤프 (예외 처리)
            return str(data)

        # 개행 제거 및 병합
        return "".join(t.replace("\n", "") for t in texts)
    except Exception as e:
        print(f"[ERROR] 예측 파일 로드 실패: {e}")
        return ""


# ======================================================================
# 3. 평가 실행 함수
# ======================================================================
def run_evaluation(gt_path: Path, pred_path: Path) -> None:
    """단일 쌍에 대한 평가를 수행하고 결과를 출력합니다."""
    
    # 파일 존재 확인
    if not gt_path.exists():
        print(f"[ERROR] GT 파일을 찾을 수 없습니다: {gt_path}")
        return
    if not pred_path.exists():
        print(f"[ERROR] 예측 파일을 찾을 수 없습니다: {pred_path}")
        return

    # 데이터 로드
    gt_text = load_gt_text(gt_path)
    pred_text = load_pred_text(pred_path)

    # Metric 계산
    cer, s, d, i = calculate_cer(gt_text, pred_text)
    distance = s + d + i
    accuracy = 1.0 - cer

    # 결과 출력
    print("=" * 60)
    print("OCR 성능 평가 결과 (Character Level)")
    print("=" * 60)
    print(f"GT 경로   : {gt_path}")
    print(f"PRED 경로 : {pred_path}")
    print("-" * 60)
    print(f"GT 길이   : {len(gt_text)}")
    print(f"PRED 길이 : {len(pred_text)}")
    print("-" * 60)
    print(f"편집 거리 (Distance) : {distance}")
    print(f" - 치환 (Substitution) : {s}")
    print(f" - 삭제 (Deletion)     : {d}")
    print(f" - 삽입 (Insertion)    : {i}")
    print("-" * 60)
    print(f"CER (Character Error Rate) : {cer:.6f}")
    print(f"문자 정확도 (Accuracy)     : {accuracy:.6f}")
    print("=" * 60)


# ======================================================================
# 4. 메인 진입점 (CLI)
# ======================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="OCR CER 성능 평가 도구")
    
    parser.add_argument(
        "--gt", 
        type=str, 
        help="정답(GT) 텍스트 파일 경로 (.txt)"
    )
    parser.add_argument(
        "--pred", 
        type=str, 
        help="예측 결과 JSON 파일 경로 (.json)"
    )

    args = parser.parse_args()

    # CLI 인자가 있으면 우선 사용, 없으면 Config 기본값 사용
    if args.gt:
        gt_path = Path(args.gt)
    else:
        gt_path = Config.DEFAULT_GT_PATH

    if args.pred:
        pred_path = Path(args.pred)
    else:
        pred_path = Config.DEFAULT_PRED_PATH

    print("[INFO] 평가 프로세스 시작...")
    run_evaluation(gt_path, pred_path)


if __name__ == "__main__":
    main()