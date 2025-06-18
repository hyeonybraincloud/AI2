!pip install mediapipe    # Mediapipe 설치

import mediapipe as mp                                  # 손 랜드마크 오버레이 목적
import os
import pathlib
import time
from tqdm import tqdm                                   # 진행률 표시
import cv2                                              # OpenCV: 이미지 처리
import torch
import torchvision as tv                                # PyTorch 비전 관련 기능
from torch import nn
from torchvision import transforms as T                 # 이미지 전처리
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler              # Mixed Precision 학습
import numpy as np
from PIL import Image                                   # 이미지 입출력
from google.colab import drive                          # Google Drive 마운트

drive.mount('/content/drive')
!tar -C /content/drive/MyDrive -cf - asl_alphabet_train | tar -C /tmp -xvf -

SRC_ROOT = '/tmp/asl_alphabet_train'
AUG_ROOT = '/tmp/asl_alphabet_augmented'
DST_ROOT = '/tmp/asl_alphabet_augmented_overlayed'
NUM_AUG = 3      # 원본 이미지 당 생성할 증강본 개수

aug_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

for d in (AUG_ROOT, DST_ROOT):
    for cls in os.listdir(SRC_ROOT):
        os.makedirs(os.path.join(d, cls), exist_ok=True)

hands = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing = mp.solutions.drawing_utils    # 랜드마크 그리기 유틸리티

for cls in os.listdir(SRC_ROOT):
    src_cls = os.path.join(SRC_ROOT, cls)
    aug_cls = os.path.join(AUG_ROOT, cls)
    dst_cls = os.path.join(DST_ROOT, cls)

    for fn in tqdm(os.listdir(src_cls), desc=f"Aug+Overlay {cls}"):
        src_path = os.path.join(src_cls, fn)
        img_pil = Image.open(src_path).convert('RGB')     # PIL 이미지로 읽기

        # 원본 포함, NUM_AUG만큼 증강본 생성
        variants = [img_pil] + [aug_transform(img_pil) for _ in range(NUM_AUG)]

        for i, img_aug in enumerate(variants):
            # PIL → OpenCV(BGR) 변환
            cv_img = cv2.cvtColor(np.array(img_aug), cv2.COLOR_RGB2BGR)
            # Mediapipe로 손 랜드마크 검출
            res = hands.process(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            # 검출된 경우, 랜드마크 및 연결선 오버레이
            if res.multi_hand_landmarks:
                drawing.draw_landmarks(
                    cv_img,
                    res.multi_hand_landmarks[0],
                    mp.solutions.hands.HAND_CONNECTIONS
                )
            # 결과 저장: 원본(_aug0), 증강본(_aug1~NUM_AUG)
            out_name = fn.replace('.jpg', f'_aug{i}.jpg')
            cv2.imwrite(os.path.join(dst_cls, out_name), cv_img)

# 리소스 해제
hands.close()

DATA_ROOT = pathlib.Path(DST_ROOT)
mean, std = [0.5]*3, [0.5]*3        # Normalize 파라미터

full_ds = tv.datasets.ImageFolder(DATA_ROOT, transform=None)

n_total = len(full_ds)
n_train = int(0.8 * n_total)
n_val   = n_total - n_train
train_ds, val_ds = random_split(
    full_ds,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_ds.dataset.transform = T.Compose([
    T.Resize((224,224)),                    # 입력 크기 맞춤
    T.ToTensor(),
    T.Normalize(mean, std),
])
val_ds.dataset.transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

train_loader = DataLoader(
    train_ds, batch_size=128, shuffle=True,
    num_workers=8, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_ds, batch_size=128, shuffle=False,
    num_workers=8, pin_memory=True
)

print(f"▶ train samples: {len(train_ds)}, val samples: {len(val_ds)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = tv.models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, len(full_ds.classes))
model = model.to(device)

if device.type == 'cuda':
    model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scaler    = GradScaler()
EPOCHS    = 10

for epoch in range(1, EPOCHS+1):
    # --- Training ---
    model.train()
    running_loss = 0.0
    t0 = time.time()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / len(train_ds)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    correct  = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    avg_val_loss = val_loss / len(val_ds)
    val_acc = correct / len(val_ds)

    t1 = time.time()
    print(f"Epoch {epoch} ▶ "
          f"Train Loss: {avg_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {t1-t0:.1f}s")

SAVE_PATH = '/content/drive/MyDrive/asl_model_local.pth'
torch.save(model.state_dict(), SAVE_PATH)
print(f"📝 Saved model to {SAVE_PATH}")