# 고문서 OCR b-box 크기 조절 테스트
def remove_giant_boxes(symbols):
    # 넓이 중위값 10배 이상 제거
    return symbols