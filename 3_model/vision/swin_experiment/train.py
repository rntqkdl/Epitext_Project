# -*- coding: utf-8 -*-
"""
Swin Transformer V2 Training Script
======================================================================
모듈: SwinV2-Small-256 Full Fine-tuning
작성자: Epitext Project Team
작성일: 2025-12-09
======================================================================
"""

import os
import gc
import json
import logging
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import cv2
from PIL import Image
from sklearn.metrics import f1_score
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 로컬 설정 임포트
try:
    from config import Config
except ImportError:
    from .config import Config

# ======================================================================
# 로깅 설정
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ======================================================================
# 데이터셋 클래스 (Dataset)
# ======================================================================
class MemoryEfficientDataset(Dataset):
    """
    메모리 상주형 데이터셋 클래스
    
    특징:
    - NPZ 샤드 파일을 메모리에 로드하여 학습 속도 최적화
    - High-RAM 환경 권장
    """

    def __init__(
        self,
        npz_dir: str,
        npz_pattern: str,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None
    ):
        self.transform = transform
        
        # 파일 탐색
        search_pattern = os.path.join(npz_dir, npz_pattern)
        self.npz_files = sorted(glob(search_pattern))

        if not self.npz_files:
            # 경로가 없어도 중단하지 않고 경고만 출력 (사용자가 경로를 확인하게 함)
            logger.warning(f"[DATASET] No NPZ files found in {search_pattern}")
            self.images = np.array([], dtype=np.uint8)
            self.labels = np.array([], dtype=np.int64)
            return

        logger.info(f"[DATASET] Found {len(self.npz_files)} NPZ files")
        self._load_data_into_memory(max_samples)
        self._analyze_class_distribution()

    def _load_data_into_memory(self, max_samples: Optional[int]):
        """전체 데이터를 메모리에 로드"""
        # 1. 샘플 수 추정
        sample_counts = []
        for npz_path in self.npz_files[:min(5, len(self.npz_files))]:
            try:
                with np.load(npz_path, mmap_mode="r") as data:
                    sample_counts.append(len(data["labels"]))
            except Exception as e:
                logger.warning(f"Failed to read {npz_path}: {e}")

        if not sample_counts:
            avg_samples = 1000
        else:
            avg_samples = int(np.mean(sample_counts))
            
        estimated_total = avg_samples * len(self.npz_files)
        
        # 2. 첫 파일로 형상 파악
        first_npz = np.load(self.npz_files[0], mmap_mode="r")
        img_shape = first_npz["images"].shape[1:]
        
        self.images = np.empty((estimated_total, *img_shape), dtype=np.uint8)
        self.labels = np.empty(estimated_total, dtype=np.int64)
        
        idx = 0
        logger.info("[DATASET] Loading data into memory...")
        
        for i, npz_path in enumerate(tqdm(self.npz_files, desc="Loading NPZ")):
            try:
                with np.load(npz_path) as data:
                    imgs = data["images"]
                    lbls = data["labels"]
                    n = len(lbls)
                    
                    if idx + n > len(self.images):
                        new_size = idx + n + avg_samples * 5
                        self.images.resize((new_size, *img_shape))
                        self.labels.resize(new_size)

                    self.images[idx:idx+n] = imgs
                    self.labels[idx:idx+n] = lbls
                    idx += n
            except Exception as e:
                logger.error(f"Error loading {npz_path}: {e}")
            
            if (i + 1) % 5 == 0:
                gc.collect()

        self.images = self.images[:idx]
        self.labels = self.labels[:idx]

        if max_samples and max_samples < len(self.labels):
            logger.warning(f"[DATASET] Limiting samples to {max_samples}")
            self.images = self.images[:max_samples]
            self.labels = self.labels[:max_samples]

        logger.info(f"[DATASET] Total loaded samples: {len(self.labels):,}")

    def _analyze_class_distribution(self):
        """클래스 분포 분석"""
        self.class_counts = Counter(self.labels.tolist())
        logger.info(f"[DATASET] Unique classes: {len(self.class_counts)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        label = self.labels[idx]

        # Grayscale -> RGB 변환
        if len(img.shape) == 2 or img.shape[-1] == 1:
            if len(img.shape) == 3:
                img = img.squeeze(-1)
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


# ======================================================================
# 데이터 전처리 및 증강 (Transforms)
# ======================================================================
def apply_histogram_equalization(img: Image.Image) -> Image.Image:
    """히스토그램 균등화 적용"""
    img_np = np.array(img)
    if len(img_np.shape) == 2:
        img_eq = cv2.equalizeHist(img_np)
    else:
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img_eq)

def get_transforms(mode: str = "train") -> transforms.Compose:
    """모드별 Transform 파이프라인 반환"""
    img_size = Config.MODEL["img_size"]
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if mode == "train":
        aug = Config.AUGMENTATION
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(aug["random_rotation"]),
            transforms.RandomAffine(
                degrees=aug["random_affine"]["degrees"],
                translate=aug["random_affine"]["translate"],
                scale=aug["random_affine"]["scale"]
            ),
            transforms.ColorJitter(
                brightness=aug["color_jitter"]["brightness"],
                contrast=aug["color_jitter"]["contrast"],
                saturation=aug["color_jitter"]["saturation"],
                hue=aug["color_jitter"]["hue"]
            ),
            transforms.RandomApply(
                [transforms.Lambda(lambda x: apply_histogram_equalization(x))],
                p=aug["histogram_equalize_prob"]
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])


# ======================================================================
# 유틸리티 함수 (Utils)
# ======================================================================
def compute_class_weights(class_counts: Dict[int, int]) -> torch.Tensor:
    """클래스 불균형 해결을 위한 가중치 계산"""
    num_classes = Config.MODEL["num_classes"]
    power = Config.LOSS["class_weight_power"]
    min_clip = Config.LOSS["min_clip"]

    counts = np.zeros(num_classes, dtype=np.float32)
    for cls_idx, count in class_counts.items():
        if cls_idx < num_classes:
            counts[cls_idx] = count

    counts = np.clip(counts, min_clip, None)
    weights = 1.0 / (counts ** power)
    weights = weights / weights.mean()
    return torch.from_numpy(weights).float()

def create_model(device: torch.device) -> nn.Module:
    """모델 생성"""
    model = timm.create_model(
        Config.MODEL["name"],
        pretrained=Config.MODEL["pretrained"],
        num_classes=Config.MODEL["num_classes"],
        img_size=Config.MODEL["img_size"]
    )
    return model.to(device)


# ======================================================================
# 학습 및 검증 엔진 (Engine)
# ======================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, writer):
    """1 Epoch 학습 수행"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    grad_accum = Config.TRAINING["grad_accumulation"]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=Config.TRAINING["use_amp"]):
            outputs = model(images)
            loss = criterion(outputs, labels) / grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss / (batch_idx + 1):.4f}",
            "acc": f"{100. * correct / total:.2f}%"
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    global_step = epoch * len(dataloader)
    writer.add_scalar("Train/Loss", epoch_loss, global_step)
    writer.add_scalar("Train/Accuracy", epoch_acc, global_step)

    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, writer):
    """검증 수행"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({
            "loss": f"{running_loss / (len(all_labels) / labels.size(0)):.4f}",
            "acc": f"{100. * correct / total:.2f}%"
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    global_step = epoch * len(dataloader)
    writer.add_scalar("Val/Loss", epoch_loss, global_step)
    writer.add_scalar("Val/Accuracy", epoch_acc, global_step)
    writer.add_scalar("Val/Macro_F1", macro_f1, global_step)

    return epoch_loss, epoch_acc, macro_f1


# ======================================================================
# 메인 실행 함수 (Main)
# ======================================================================
def main():
    Config.print_config()
    
    # 0. 디렉토리 생성
    Path(Config.DATA["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(Config.DATA["tensorboard_dir"]).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. 데이터셋 준비
    logger.info("[STEP 1] Loading Dataset...")
    train_dataset = MemoryEfficientDataset(
        Config.DATA["base_dir"], 
        Config.DATA["train_dir"],
        Config.DATA["train_pattern"],
        transform=get_transforms("train")
    )
    val_dataset = MemoryEfficientDataset(
        Config.DATA["base_dir"], 
        Config.DATA["val_dir"],
        Config.DATA["val_pattern"],
        transform=get_transforms("val")
    )
    
    if len(train_dataset) == 0:
        logger.error("Dataset is empty. Check your paths in config.py")
        return

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.TRAINING["batch_size"], 
        shuffle=True, 
        num_workers=Config.TRAINING["num_workers"], 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.TRAINING["batch_size"], 
        shuffle=False, 
        num_workers=Config.TRAINING["num_workers"]
    )

    # 2. 모델 및 학습 요소 설정
    logger.info("[STEP 2] Initializing Model & Optimizer...")
    class_weights = compute_class_weights(train_dataset.class_counts).to(device)
    model = create_model(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    head_params = [p for n, p in model.named_parameters() if "head" in n]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": Config.TRAINING["lr_backbone"]},
        {"params": head_params, "lr": Config.TRAINING["lr_head"]}
    ], weight_decay=Config.TRAINING["weight_decay"])

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=Config.TRAINING["warmup_epochs"])
    cosine = CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.TRAINING["cosine_t0"], T_mult=Config.TRAINING["cosine_tmult"]
    )
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[Config.TRAINING["warmup_epochs"]])
    scaler = GradScaler()

    # 3. 학습 루프
    run_name = f"swin_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(Config.DATA["tensorboard_dir"], run_name))
    
    best_f1 = 0.0
    patience = 0
    
    logger.info("[STEP 3] Start Training...")
    
    for epoch in range(Config.TRAINING["num_epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, writer
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        
        scheduler.step()
        logger.info(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")

        # 체크포인트 저장 로직 (간소화)
        ckpt_path = Path(Config.DATA["output_dir"]) / f"ckpt_epoch_{epoch+1}.pth"
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            torch.save(model.state_dict(), str(ckpt_path).replace(".pth", "_best.pth"))
        else:
            patience += 1
            
        if patience >= Config.TRAINING["early_stopping_patience"]:
            logger.info("Early stopping triggered.")
            break

    writer.close()
    logger.info("Training Finished.")

if __name__ == "__main__":
    main()