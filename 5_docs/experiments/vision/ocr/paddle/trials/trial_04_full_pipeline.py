# Paddle OCR 시행착오-4: Full Pipeline V2
# 세로 읽기 순서 고정 (오른->왼) + 한자 마스킹
from paddleocr import PaddleOCR
def order_vertical_boxes(boxes):
    # X좌표 기준 정렬
    return boxes