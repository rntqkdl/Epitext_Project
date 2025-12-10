# CRAFT를 이용한 한자 인식
# 목적: CRAFT 탐지기로 한자 영역을 검출하고 뭉텅이를 watershed로 분할
# 요약: CRAFT 버전 호환을 감안해 detect_text를 호출하고, 큰 영역은 거리변환+watershed로 나눠 작은 글자를 추출
# 작성일: 2025-12-10
# -*- coding: utf-8 -*-
"""
CRAFT ()  +  (watershed) 
- craft-text-detector    
-  ( ) +watershed   
- :  PNG,  JSON, ()  
"""
import json
import inspect
from pathlib import Path
import cv2
import numpy as np
from craft_text_detector import Craft

IMG_PATH = r"C:\Users\myjew\takbon\test5.png"
OUT_DIR  = r"C:\hanja_craft_out"
SAVE_CROPS = True
PARAMS = dict(
    text_threshold=0.72,
    low_text=0.40,
    link_threshold=0.20,
    long_size=1920,
    cuda=False,
    refiner=False,
)
MIN_BOX_W, MIN_BOX_H = 8, 8
SPLIT_ENABLE = True
AREA_FACTOR = 3.0
SPLIT_MIN_CHAR_PIXELS = 8
SPLIT_MAX_CHARS = 32

def sort_poly_clockwise(poly):
    poly = np.array(poly, dtype=np.float32)
    c = np.mean(poly, axis=0)
    ang = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
    idx = np.argsort(ang)
    return poly[idx].tolist()

def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    return [x1, y1, int(x2 - x1), int(y2 - y1)]

def as_list(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def run_craft_detect(img_path: str, params: dict):
    craft = Craft(cuda=params.get("cuda", False), refiner=params.get("refiner", False))
    sig = inspect.signature(craft.detect_text)
    accepts = set(sig.parameters.keys())
    if {"text_threshold", "low_text", "link_threshold", "long_size"}.issubset(accepts):
        prediction = craft.detect_text(
            img_path,
            text_threshold=params["text_threshold"],
            low_text=params["low_text"],
            link_threshold=params["link_threshold"],
            long_size=params["long_size"],
        )
        return craft, prediction
    craft = Craft(
        cuda=params.get("cuda", False),
        refiner=params.get("refiner", False),
        text_threshold=params.get("text_threshold", 0.7),
        low_text=params.get("low_text", 0.4),
        link_threshold=params.get("link_threshold", 0.4),
        long_size=params.get("long_size", 1280),
    )
    prediction = craft.detect_text(img_path)
    return craft, prediction

def split_blob_into_chars(crop_bgr, min_char=8, max_chars=16):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, k, iterations=1)
    dist = cv2.distanceTransform(binv, cv2.DIST_L2, 3)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    _, peaks = cv2.threshold((dist_norm * 255).astype(np.uint8), 120, 255, cv2.THRESH_BINARY)
    peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, k, iterations=1)
    n_labels, markers = cv2.connectedComponents(peaks)
    markers = markers + 1
    markers[binv == 0] = 0
    ws_in = cv2.cvtColor(binv, cv2.COLOR_GRAY2BGR)
    cv2.watershed(ws_in, markers)
    boxes = []
    for lab in range(2, n_labels + 1):
        mask = (markers == lab).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < min_char:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        if w >= min_char and h >= min_char:
            boxes.append((x, y, w, h))
    if len(boxes) > max_chars:
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:max_chars]
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes

def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlay"; overlay_dir.mkdir(exist_ok=True)
    crops_dir = out_dir / "crops"; crops_dir.mkdir(exist_ok=True)
    img_bgr = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"   : {IMG_PATH}"
    H, W = img_bgr.shape[:2]
    craft = None
    try:
        craft, prediction = run_craft_detect(IMG_PATH, PARAMS)
        polys = as_list(prediction.get("polys"))
        boxes = as_list(prediction.get("boxes"))
        regions = polys if len(polys) > 0 else boxes
        raw = []
        areas = []
        for region in regions:
            poly = sort_poly_clockwise(region)
            x, y, bw, bh = poly_to_bbox(poly)
            if bw < MIN_BOX_W or bh < MIN_BOX_H:
                continue
            raw.append((poly, x, y, bw, bh))
            areas.append(bw * bh)
        median_area = float(np.median(areas)) if areas else 0.0
        results = []
        vis = img_bgr.copy()
        for poly, x, y, bw, bh in raw:
            area = bw * bh
            split_done = False
            if SPLIT_ENABLE and median_area > 0 and area >= AREA_FACTOR * median_area:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + bw), min(H, y + bh)
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    char_boxes = split_blob_into_chars(
                        crop,
                        min_char=SPLIT_MIN_CHAR_PIXELS,
                        max_chars=SPLIT_MAX_CHARS
                    )
                    if len(char_boxes) >= 2:
                        split_done = True
                        for (cx, cy, cw, ch) in char_boxes:
                            gx, gy, gw, gh = x + cx, y + cy, cw, ch
                            if gw < MIN_BOX_W or gh < MIN_BOX_H:
                                continue
                            results.append({
                                "index": len(results) + 1,
                                "poly": [[gx, gy], [gx+gw, gy], [gx+gw, gy+gh], [gx, gy+gh]],
                                "bbox": [int(gx), int(gy), int(gw), int(gh)],
                                "split": True
                            })
                            cv2.rectangle(vis, (gx, gy), (gx+gw, gy+gh), (0, 200, 255), 2)
                            cv2.putText(vis, f"{len(results)}", (gx, max(0, gy-5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                            if SAVE_CROPS:
                                gx1, gy1 = max(0, gx), max(0, gy)
                                gx2, gy2 = min(W, gx+gw), min(H, gy+gh)
                                gcrop = img_bgr[gy1:gy2, gx1:gx2]
                                if gcrop.size > 0:
                                    cv2.imwrite(str(crops_dir / f"char_{len(results):04d}.png"), gcrop)
            if not split_done:
                results.append({
                    "index": len(results) + 1,
                    "poly": [[int(px), int(py)] for px, py in poly],
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "split": False
                })
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 0, 0), 1)
                cv2.putText(vis, f"{len(results)}", (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                if SAVE_CROPS:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(W, x + bw), min(H, y + bh)
                    crop = img_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(str(crops_dir / f"blob_{len(results):04d}.png"), crop)
        base = Path(IMG_PATH).stem
        cv2.imwrite(str((Path(OUT_DIR) / "overlay" / f"{base}_overlay.png")), vis)
        with open(Path(OUT_DIR) / f"{base}_boxes.json", "w", encoding="utf-8") as f:
            json.dump({
                "image": str(Path(IMG_PATH)),
                "size": {"w": W, "h": H},
                "count": len(results),
                "median_area": median_area,
                "area_factor": AREA_FACTOR,
                "split_enable": SPLIT_ENABLE,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print(f"[]  {len(results)}  ( )")
        print(f" - : {Path(OUT_DIR) / 'overlay' / (base + '_overlay.png')}")
        print(f" -  JSON: {Path(OUT_DIR) / (base + '_boxes.json')}")
        if SAVE_CROPS:
            print(f" -  : {Path(OUT_DIR) / 'crops'}")
    finally:
        if craft is not None:
            try: craft.unload_craftnet_model()
            except Exception: pass
            try: craft.unload_refinenet_model()
            except Exception: pass

if __name__ == "__main__":
    main()
