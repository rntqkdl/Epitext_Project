"""
NLP Preprocessing Configuration
======================================================================
목적: 텍스트 전처리에 필요한 파일 경로, 정규식 패턴, 파라미터 관리
작성자: Epitext Project Team
======================================================================
"""

import os
import re
from pathlib import Path

class Config:
    """전역 설정 클래스"""
    
    # ==================================================================
    # 1. 경로 설정
    # ==================================================================
    # 현재 파일 기준 프로젝트 루트 경로 (1_data/preprocess/nlp -> 1_data)
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    
    # 입력 데이터 경로 (크롤링된 원본 CSV)
    INPUT_CSV = BASE_DIR / "raw_data" / "doc_id_transcript_dataset.csv"
    
    # 출력 데이터 경로 (전처리 완료된 CSV)
    OUTPUT_CSV = BASE_DIR / "raw_data" / "doc_id_transcript_dataset_processed.csv"
    
    # ==================================================================
    # 2. 전처리 파라미터
    # ==================================================================
    # 최소 텍스트 길이 (이보다 짧으면 삭제)
    MIN_LENGTH = 20
    
    # 제거할 노이즈 키워드 목록
    NOISE_KEYWORDS = [
        "譯註", "韓國金石", "高麗靑瓷", "調査報告",
        "側銘", "小口銘", "金石文", "全文", "遺文"
    ]
    
    # 노이즈 패턴 컴파일 (속도 최적화)
    NOISE_PATTERN = re.compile("|".join(re.escape(k) for k in NOISE_KEYWORDS))
    
    # 제거할 특수문자 패턴 (한자, 숫자 외의 잡음 제거)
    CHARS_TO_REMOVE = (
        r"[ㄱ-ㅎ가-힣0-9\[\]\(\)【】〔〕『』｢｣·,…\?\/\:''「․（）［］\.〈〉◎;!"
        r"\"\'\*\+@\{\}""⃞▩、《》ㅜㅡㅣㆍ"
        r"#%&<=>ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        r"\\_`abcdefghijklmnopqrstuvwxyz|~"
        r"€×íāīŚśū˅᠁ṃṅṇṣṭ"
        r"–—―‥※ⅠⅡⅢ→↩∕∘∙∧∨∴⊏⊐⊙⎀"
        r"①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑴⑵"
        r"─━│┃┌┎┏┒┓└┗┘┛├┠┤┥┲╋\u25a3\u2614\u274d\u2b1c"
        r"おけこしせそてとにのみるを"
        r"アイシジスセタテデトナニノハホマリヲ・"
        r"㈠㈡㈢㉠㉡㉢㉣㉤㉥"
        r"」\u2606"
        r"\u200b\x80\ufeff"
        r"-]"
    )
    
    # 판독 불가 기호 마스킹 패턴 (-> ▨)
    SYMBOLS_TO_REPLACE = r"\(판독불가\)|[▦▧△▼◆○◦◯\u2610□■？]"
    
    @staticmethod
    def print_config():
        """설정 정보 출력"""
        print("======================================================")
        print(" NLP Preprocessing Configuration")
        print("======================================================")
        print(f" Input Path:  {Config.INPUT_CSV}")
        print(f" Output Path: {Config.OUTPUT_CSV}")
        print(f" Min Length:  {Config.MIN_LENGTH}")
        print("======================================================")