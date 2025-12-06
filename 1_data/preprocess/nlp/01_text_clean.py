"""
텍스트 전처리 - 노이즈 제거 및 필터링
======================================================================
작성자: 4조 복원왕 김탁본
작성일: 2025-12-07
출처: 4주차 보고서
======================================================================
"""

import pandas as pd
import re
import os
from tqdm import tqdm


# ======================================================================
# 설정
# ======================================================================
INPUT_CSV = "doc_id_transcript_dataset.csv"
OUTPUT_CSV = "doc_id_transcript_dataset_processed.csv"
MIN_LENGTH = 20

NOISE_KEYWORDS = [
    "譯註", "韓國金石", "高麗靑瓷", "調査報告",
    "側銘", "小口銘", "金石文", "全文", "遺文"
]
noise_pattern = re.compile("|".join(re.escape(k) for k in NOISE_KEYWORDS))


# ======================================================================
# 전처리 함수
# ======================================================================
def clean_text_base(line_text):
    """
    기본 텍스트 세정
    - 판독불가 기호 마스킹
    - 특수문자 제거
    - 노이즈 키워드 제거
    """
    if not isinstance(line_text, str):
        return ""

    try:
        # 마스킹 토큰 처리
        symbols_to_replace = r"\(판독불가\)|[▦▧△▼◆○◦◯☐□■？]"
        text = re.sub(symbols_to_replace, "▨", line_text)

        # 특수문자 제거
        chars_to_remove = (
            r"[ㄱ-ㅎ가-힣0-9\[\]\(\)【】〔〕『』｢｣·,…\?\/\:''「․（）［］\.〈〉◎;!"
            r"\"\'\*\+@\{\}""⃞▩、《》ㅜㅡㅣㆍ"
            r"#%&<=>ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            r"\\_`abcdefghijklmnopqrstuvwxyz|~"
            r"€×íāīŚśū˅᠁ṃṅṇṣṭ"
            r"–—―‥※ⅠⅡⅢ→↩∕∘∙∧∨∴⊏⊐⊙⎀"
            r"①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑴⑵"
            r"─━│┃┌┎┏┒┓└┗┘┛├┠┤┥┲╋▣☔❍⬜"
            r"おけこしせそてとにのみるを"
            r"アイシジスセタテデトナニノハホマリヲ・"
            r"㈠㈡㈢㉠㉡㉢㉣㉤㉥"
            r"」☆"
            r"\u200b\x80\ufeff"
            r"-]"
        )
        text = re.sub(chars_to_remove, "", text)

        # 노이즈 키워드 제거
        text = noise_pattern.sub("", text)

        # 공백 정리
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = text.strip()

        return text
    except Exception:
        return ""


def flatten_text(text):
    """줄바꿈 제거 및 공백 정리"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


# ======================================================================
# 메인 실행
# ======================================================================
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"파일을 찾을 수 없습니다: {INPUT_CSV}")
        return

    print(f"데이터 로드 중: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    original_count = len(df)

    print(f"전처리 시작 (총 {original_count} 행)...")

    tqdm.pandas(desc="Preprocessing (Clean & Flatten)")
    df["preprocess"] = df["transcript"].progress_apply(
        lambda x: flatten_text(clean_text_base(x))
    )

    print(f"\n[필터링] preprocess 길이 {MIN_LENGTH}자 미만 삭제 중...")
    df_filtered = df[df["preprocess"].str.len() >= MIN_LENGTH].copy()
    removed_count = original_count - len(df_filtered)

    print(f"-> 삭제됨: {removed_count}개")
    print(f"-> 남음: {len(df_filtered)}개")

    print("\n--- 결과 데이터 샘플 (첫 번째 행) ---")
    if not df_filtered.empty:
        sample = df_filtered.iloc[0]
        print(f"[ID] {sample['doc_id']}")
        print(f"[preprocess]:\n{sample['preprocess'][:100]}...")
    else:
        print("남은 데이터가 없습니다.")

    df_filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUTPUT_CSV}")
    print(f"생성된 컬럼: {list(df_filtered.columns)}")


if __name__ == "__main__":
    main()
