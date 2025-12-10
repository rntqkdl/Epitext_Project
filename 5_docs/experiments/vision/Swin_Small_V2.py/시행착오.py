# Swin Small V2 Production Experiment
# 목적: 한자 탁본 이미지를 분류하기 위한 SwinV2_small 모델 학습 코드
# 요약: 서체와 클래스 불균형을 고려한 가중치 계산, 메모리 효율적인 데이터셋, 학습 및 검증 루프, 조기 종료를 포함한 훈련 스크립트입니다.
# 작성일: 2025-12-10
# -*- coding: utf-8 -*-
"""
================================================================================
한자 탁본 AI A100 High-RAM PRODUCTION (완전 최적화)
================================================================================
char_mapping.json 구조 확인: ['char2idx', 'idx2char', 'total_classes', 'unique_chars']
EDA 100% 반영: 불균형/서체/밝기/Val전용
Back-end 완벽 호환
1에폭 1시간, Early Stopping 3회
================================================================================
"""

import os
import glob
import json
import warnings
import gc
import logging
from collections import Counter
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torchvision import transforms
from sklearn.metrics import f1_score
import timm
from torch.utils.tensorboard import SummaryWriter
import psutil

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*100)
print("한자 탁본 AI A100 PRODUCTION (최종 완성)")
print("="*100)

# ========================= 환경 확인 =========================
mem = psutil.virtual_memory()
print(f"RAM: {mem.total/1e9:.1f}GB ({mem.percent:.1f}% 사용)")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ========================= 경로 =========================
DRIVE_BASE = "/content/drive/MyDrive/Colab Notebooks/notion_files_backup"
LOCAL_BASE = "/content/hanja_local"
CHAR_JSON = "/content/drive/MyDrive/Colab Notebooks/char_mapping.json"
OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/hanja_a100_production"

os.makedirs(LOCAL_BASE, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = LOCAL_BASE

# ========================= 하이퍼파라미터 =========================
DEVICE = torch.device("cuda")
IMG_SIZE = 256
BATCH_SIZE = 192
NUM_EPOCHS = 50
NUM_WORKERS = 8
LR_HEAD, LR_BACKBONE = 3e-4, 3e-5
WD, ACCUM_STEPS = 0.01, 4

print(f"\n설정: Batch {BATCH_SIZE}×{ACCUM_STEPS}={BATCH_SIZE*ACCUM_STEPS}")

# ========================= 문자 매핑 (구조 확인됨) =========================
print("\nchar_mapping.json 로드...")
with open(CHAR_JSON, encoding="utf-8") as f:
    char_mapping = json.load(f)

NUM_CLASSES = char_mapping["total_classes"]
idx2char = {int(k): v for k, v in char_mapping["idx2char"].items()}

print(f"클래스: {NUM_CLASSES:,}개")
print(f"idx2char: {len(idx2char):,}개")
print(f"키 확인: {list(char_mapping.keys())}")
print()

# ========================= EDA-1&2: 서체+클래스 가중치 =========================
def compute_style_aware_weights(data_dir, num_classes):
    print("[EDA] 서체+클래스 가중치 계산...")
    train_pattern = os.path.join(data_dir, "*train_20percent_part*.npz")
    shards = sorted(glob.glob(train_pattern))[:12]

    class_counter, style_counter = Counter(), Counter()

    for path in tqdm(shards, desc="분석"):
        try:
            with np.load(path, allow_pickle=True) as z:
                labels, metas = z["labels"][:8000], z["metadata"][:8000]
                class_counter.update(labels)

                for meta in metas:
                    if hasattr(meta, 'item'):
                        meta = meta.item()
                    if isinstance(meta, dict):
                        style = meta.get('style', 'unknown')
                        style_counter[style] += 1
        except:
            continue

    print(f"클래스: {len(class_counter):,} | 서체: {dict(style_counter.most_common(3))}")

    # 클래스 가중치 (1:6072 불균형)
    counts = np.zeros(num_classes)
    for k, v in class_counter.items():
        if k < num_classes:
            counts[k] = v
    counts = np.clip(counts, 1, None)
    class_weights = 1.0 / counts

    # 서체 팩터 (해서 87.9% 편향)
    style_factor = np.mean([1.0/count for count in style_counter.values()])
    weights = class_weights * style_factor
    weights = np.clip(weights, 1.0, 100.0) / weights.mean()

    print(f"가중치 범위: {weights.min():.1f}~{weights.max():.1f}x")
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


class_weights = compute_style_aware_weights(DATA_DIR, NUM_CLASSES)

# ========================= MemoryEfficientDataset =========================
class MemoryEfficientDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.transform = transform
        pattern = os.path.join(data_dir, "*train_20percent_part*.npz") if split == "train" else os.path.join(data_dir, "*_val_20percent_part*_deg.npz")
        shards = sorted(glob.glob(pattern))

        # 크기 추정
        total = sum(len(np.load(p, allow_pickle=True)["labels"]) for p in shards[:6])
        estimated = (total // 6) * len(shards)
        img_shape = np.load(shards[0], allow_pickle=True)["images"].shape[1:]

        # 메모리 할당
        self.images = np.empty((estimated, *img_shape), dtype=np.uint8)
        self.labels = np.empty(estimated, dtype=np.int64)

        # 로딩
        idx = 0
        for path in tqdm(shards, desc=f"{split}"):
            try:
                with np.load(path, allow_pickle=True) as z:
                    n = len(z["labels"])
                    self.images[idx:idx+n] = z["images"]
                    self.labels[idx:idx+n] = z["labels"]
                    idx += n
            except:
                continue

        self.images, self.labels = self.images[:idx], self.labels[:idx]
        print(f"{split.upper()}: {len(self.labels):,}개 ({self.images.nbytes/1e9:.1f}GB)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return self.transform(Image.fromarray(img)), label

# ========================= EDA-3: 밝기 증강 (0.078 차이 완화) =========================
train_tf = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.9,1.1)),
    transforms.RandomPerspective(0.15, p=0.5),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.15, hue=0.08),
    transforms.Lambda(lambda x: ImageOps.equalize(x) if np.random.rand() < 0.4 else x),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ========================= Dataset =========================
print("\nDataset 생성...")
train_ds = MemoryEfficientDataset(DATA_DIR, "train", train_tf)
val_ds = MemoryEfficientDataset(DATA_DIR, "val", val_tf)

# ========================= EDA-4: Val 전용 클래스 보강 =========================
def balance_val_classes(train_ds, val_ds, ratio=0.03):
    print("\n[EDA-4] Val 전용 클래스 보강...")
    train_classes = set(train_ds.labels)
    val_classes = set(val_ds.labels)
    val_only = val_classes - train_classes

    print(f" Val 전용: {len(val_only):,}개")

    if len(val_only) > 0:
        val_only_idx = np.array([i for i, l in enumerate(val_ds.labels) if l in val_only])
        sample_size = max(1, int(len(val_only_idx) * ratio))
        sample_idx = np.random.choice(val_only_idx, sample_size, replace=False)

        train_ds.images = np.concatenate([train_ds.images, val_ds.images[sample_idx]])
        train_ds.labels = np.concatenate([train_ds.labels, val_ds.labels[sample_idx]])

        mask = np.ones(len(val_ds.labels), dtype=bool)
        mask[sample_idx] = False
        val_ds.images, val_ds.labels = val_ds.images[mask], val_ds.labels[mask]

        print(f"{sample_size:,}개 Train 이동")

    return train_ds, val_ds


train_ds, val_ds = balance_val_classes(train_ds, val_ds)

# ========================= DataLoader =========================
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

# ========================= Back-end 호환 모델 =========================
print("\nSwinV2_small_window16_256 (Back-end 호환)...")
model = timm.create_model("swinv2_small_window16_256", pretrained=True,
                         num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(DEVICE)

backbone_params = [p for name, p in model.named_parameters() if "head" not in name]
head_params = [p for name, p in model.named_parameters() if "head" in name]

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LR_BACKBONE, 'weight_decay': WD},
    {'params': head_params, 'lr': LR_HEAD, 'weight_decay': WD}
])

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=3)
cosine = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[3])
scaler = torch.amp.GradScaler("cuda")

# ========================= 체크포인트 =========================
start_epoch, best_f1 = 0, 0.0
ckpts = glob.glob(os.path.join(OUTPUT_DIR, "best_model_f1_*.pth"))
if ckpts:
    latest = max(ckpts, key=os.path.getmtime)
    try:
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt['macro_f1']
        print(f"재시작: E{start_epoch} (F1: {best_f1:.4f})")
    except:
        pass

# ========================= 학습 함수 =========================
def train_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Train E{epoch+1}", ncols=120)

    optimizer.zero_grad(set_to_none=True)
    for i, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda"):
            outs = model(imgs)
            loss = criterion(outs, labels)

        scaler.scale(loss / ACCUM_STEPS).backward()
        if (i + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        correct += outs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f"{total_loss/(i+1):.3f}",
            'acc': f"{100*correct/total:.1f}%"
        })

    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, epoch):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, correct, total = 0, 0, 0
    batch_count = 0

    pbar = tqdm(loader, desc=f"Val E{epoch+1}", ncols=120)
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outs = model(imgs)
                loss = criterion(outs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outs.argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_count += 1
            acc = 100.0 * correct / total

            if batch_count % 30 == 0:
                recent_f1 = f1_score(all_labels[-2000:], all_preds[-2000:], average='macro', zero_division=0)
                pbar.set_postfix({'acc': f"{acc:.1f}%", 'f1': f"{recent_f1:.3f}"})
            else:
                pbar.set_postfix({'acc': f"{acc:.1f}%"})

    avg_loss = total_loss / total
    val_acc = 100.0 * correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\nE{epoch+1}: Loss={avg_loss:.4f} | Acc={val_acc:.2f}% | F1={macro_f1:.4f}")
    return avg_loss, val_acc, macro_f1


# ========================= 메인 루프 =========================
writer = SummaryWriter(os.path.join(OUTPUT_DIR, "logs"))
patience = 0
PATIENCE_MAX = 3

print("\n" + "="*100)
print(f"학습 시작! E{start_epoch+1}-{NUM_EPOCHS}")
print(f" Train: {len(train_ds):,} | Val: {len(val_ds):,}")
print("EDA 전부 반영 완료!")
print("="*100)

for epoch in range(start_epoch, NUM_EPOCHS):
    tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch)
    val_loss, val_acc, macro_f1 = validate(model, val_loader, criterion, epoch)

    writer.add_scalar("Loss/train", tr_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Acc/train", tr_acc, epoch)
    writer.add_scalar("Acc/val", val_acc, epoch)
    writer.add_scalar("F1/macro", macro_f1, epoch)

    scheduler.step()

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        patience = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'macro_f1': macro_f1,
            'val_accuracy': val_acc,
            'train_accuracy': tr_acc,
            'num_classes': NUM_CLASSES,
            'char_mapping': {'idx2char': idx2char},
            'class_weights': class_weights.cpu()
        }
        save_path = os.path.join(OUTPUT_DIR, f"best_model_f1_{macro_f1:.4f}_E{epoch+1}.pth")
        torch.save(checkpoint, save_path)
        print(f"NEW BEST! F1={macro_f1:.4f} → {os.path.basename(save_path)}")
    else:
        patience += 1
        print(f"Patience: {patience}/{PATIENCE_MAX}")

    if patience >= PATIENCE_MAX:
        print(f"\nEarly Stopping E{epoch+1} (최종 F1: {best_f1:.4f})")
        break

    gc.collect()
    torch.cuda.empty_cache()

writer.close()

print("\n" + "="*100)
print("학습 완료!")
print(f"최종 BEST F1: {best_f1:.4f}")
print(f"저장: {OUTPUT_DIR}")
print(f"TensorBoard: %tensorboard --logdir {os.path.join(OUTPUT_DIR, 'logs')}")
print("\nBack-end에서 바로 사용 가능!")
print("="*100)