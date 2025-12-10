# 1차 저장: 세로읽기 기본 로직
import os, io
from google.cloud import vision
def normalize_box_sizes(symbols):
    # 중앙값 기반 박스 크기 정규화
    return symbols