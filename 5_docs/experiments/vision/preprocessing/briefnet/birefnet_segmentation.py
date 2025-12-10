# briefnet을 이용한 이미지 전처리
# 목적: BiRefNet 등 세그멘테이션 모델로 전경을 분리하여 배경을 제거
# 요약: 딥러닝 세그멘테이션을 이용해 한자 영역만 분리하고 마스크를 이용한 배경제거를 구현
# 작성일: 2025-12-10
from transformers import AutoModelForImageSegmentation, AutoImageProcessor
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def run_birefnet_segmentation(img_path: str, model_id: str = "Zhengpeng7/BiRefNet"):
    processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    img_path_obj = Path(img_path)
    if img_path_obj.is_file():
        img = Image.open(img_path_obj).convert("RGB")
    else:
        # fallback: download a demo image
        demo_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-image-classification/resolve/main/imagenet_classification/000000039769.png"
        img = Image.open(requests.get(demo_url, stream=True).raw).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, (list, tuple)):
        logits = outputs[0]
    else:
        logits = outputs
    if logits.shape[1] == 1:
        pred_mask = torch.sigmoid(logits[0, 0])
    else:
        probs = torch.softmax(logits[0], dim=0)
        fg_class = 1
        pred_mask = probs[fg_class]
    pred_mask_np = pred_mask.detach().cpu().numpy()
    mask_bin = (pred_mask_np > 0.5).astype(np.uint8)
    img_np = np.array(img)
    mask_3ch = np.repeat(mask_bin[..., None], 3, axis=2)
    white_bg = np.ones_like(img_np, dtype=np.uint8) * 255
    seg_result = np.where(mask_3ch == 1, img_np, white_bg)
    return img_np, pred_mask_np, seg_result
