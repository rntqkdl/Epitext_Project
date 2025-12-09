"""
Swin Transformer Evaluation Script
======================================================================
목적: 학습된 모델 로드, Top-K 예측, 결과 시각화 (텍스트 겹침 해결 버전)
작성자: Epitext Project Team
======================================================================
"""

import os
import sys
import logging
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# 로컬 설정 임포트 시도
try:
    from config import Config
except ImportError:
    from .config import Config

# ======================================================================
# 한자 폰트 및 로거 설정
# ======================================================================
plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("SwinEval")


# ======================================================================
# Swin 엔진 클래스
# ======================================================================
class SwinEngine:
    """Swin Transformer 모델 추론 엔진"""
    
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading Swin model...")
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
            
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.num_classes = ckpt["num_classes"]
        
        # 인덱스 -> 문자 매핑 복원
        self.idx2char = {
            int(k): v for k, v in ckpt["char_mapping"].get("idx2char", {}).items()
        }
        
        # 모델 생성 및 가중치 로드
        self.model = timm.create_model(
            Config.MODEL_NAME, 
            pretrained=False, 
            num_classes=self.num_classes, 
            img_size=Config.IMG_SIZE
        ).to(self.device)
        
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
        # 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        logger.info(f"Model ready (Classes: {self.num_classes}, Device: {self.device})")

    def predict_top_k(self, img_rgb, k=5):
        """단일 이미지에 대한 Top-K 예측 수행"""
        pil_image = Image.fromarray(img_rgb)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            
        top_probs, top_indices = torch.topk(probs, k)
        
        return [
            {"token": self.idx2char.get(int(idx), "?"), "prob": float(prob)} 
            for prob, idx in zip(top_probs, top_indices)
        ]


# ======================================================================
# 유틸리티 함수
# ======================================================================
def load_npz_batch(npz_path):
    """NPZ 파일에서 배치 단위 데이터 로드"""
    if not os.path.exists(npz_path):
        logger.warning(f"File not found: {npz_path}")
        return np.array([]), np.array([])
        
    data = np.load(npz_path)
    if Config.IMAGES_KEY not in data or Config.LABELS_KEY not in data:
        logger.warning(f"Keys missing in NPZ: {list(data.keys())}")
        return np.array([]), np.array([])
    
    images = data[Config.IMAGES_KEY]
    labels = data[Config.LABELS_KEY].astype(int)
    
    # 랜덤 샘플링
    N = min(Config.BATCH_SIZE, len(images))
    indices = np.random.choice(len(images), N, replace=False)
    
    return images[indices], labels[indices]

def normalize_to_rgb(img):
    """이미지를 RGB 포맷으로 정규화"""
    if img.ndim == 2:
        img = np.stack([img, img, img], -1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, 2)
        
    if img.dtype != np.uint8:
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
            
    return img


# ======================================================================
# 평가 및 시각화 함수
# ======================================================================
def run_test(engine):
    """모델 평가 및 결과 시각화 실행"""
    if not Config.NPZ_PATHS:
        logger.error("No NPZ paths defined in Config.")
        return

    npz_path = Config.NPZ_PATHS[0]
    images, labels = load_npz_batch(npz_path)
    
    if len(images) == 0:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Testing {len(images)} samples...")
    
    viz_samples = []
    top1_correct, total = 0, 0
    
    # 추론 루프
    for i in range(min(20, len(images))):
        img, label = images[i], labels[i]
        rgb = normalize_to_rgb(img)
        preds = engine.predict_top_k(rgb)
        
        true_char = engine.idx2char.get(label, "?")
        pred_top1 = preds[0]["token"]
        
        total += 1
        if pred_top1 == true_char:
            top1_correct += 1
        
        if len(viz_samples) < Config.NUM_VIZ_TOP1:
            viz_samples.append({
                "image": rgb, 
                "true": true_char, 
                "pred": pred_top1,
                "top5": preds[:5], 
                "correct": pred_top1 == true_char
            })
    
    # 정확도 출력
    acc = top1_correct / total
    print(f"\n Top-1 Accuracy: {acc:.2%} ({top1_correct}/{total})")
    
    # 샘플 상세 출력
    print("\n SAMPLE DETAILS:")
    for i, sample in enumerate(viz_samples, 1):
        print(f"\nSample {i}:")
        print(f"  True: {sample['true']}")
        print(f"  Pred: {sample['pred']} ({sample['top5'][0]['prob']:.1%})")
        print(f"  {'[Correct]' if sample['correct'] else '[Wrong]'}")
    
    # 시각화 (텍스트 분리 버전)
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(10, 8),
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("SwinV2 Hanja Top-1 Test", fontsize=18, fontweight="bold")

    for col, sample in enumerate(viz_samples):
        img_ax = axes[0, col]   # 위쪽: 이미지
        txt_ax = axes[1, col]   # 아래쪽: 텍스트

        # 1) 이미지
        img_ax.imshow(sample["image"])
        img_ax.axis("off")
        img_ax.set_title(f"Sample {col+1}", fontsize=14, pad=5, fontweight="bold")

        # 2) 텍스트
        txt_ax.axis("off")

        true_txt = f"True: {sample['true']}"
        pred_txt = f"Pred: {sample['pred']} ({sample['top5'][0]['prob']:.1%})"
        status_txt = "CORRECT" if sample["correct"] else "WRONG"
        status_color = "green" if sample["correct"] else "red"

        txt_ax.text(0.5, 0.75, true_txt, ha="center", va="center", fontsize=14, fontweight="bold")
        txt_ax.text(0.5, 0.45, pred_txt, ha="center", va="center", fontsize=12)
        txt_ax.text(0.5, 0.15, status_txt, ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white", bbox=dict(boxstyle="round,pad=0.4", facecolor=status_color, alpha=0.8))

    plt.tight_layout()
    save_path = Config.OUTPUT_DIR / "swin_top1_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    
    logger.info(f"Test completed! Result saved to: {save_path}")


# ======================================================================
# 메인 함수
# ======================================================================
def main():
    Config.print_config()
    print("======================================================")
    print(" SwinV2 Hanja Classifier - Top-1 Test")
    print("======================================================")
    
    engine = SwinEngine(Config.CHECKPOINT_PATH)
    run_test(engine)

if __name__ == "__main__":
    main()