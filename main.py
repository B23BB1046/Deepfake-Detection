import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b4
from sklearn.metrics import f1_score, classification_report, precision_recall_curve
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

metadata_path = os.path.join(PROJECT_ROOT, "csv", "FF++_Metadata.csv")
if os.path.exists(metadata_path):
    df = pd.read_csv(metadata_path)
    print(df.head())
    print(df.columns)
    print(df.shape)
else:
    print(f"[INFO] Metadata not found: {metadata_path}")

def mixup_data(x, y, alpha=0.25):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def get_train_transform(img_size=380):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),                     
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),  # softened
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)),  # softened
        # Gaussian blur removed - blurs subtle deepfake artifacts you actually need to detect
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.01, 0.05)),  # softened
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_val_transform(img_size=380):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "val")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")

train_dataset = ImageFolder(root=TRAIN_DIR, transform=get_train_transform())
val_dataset = ImageFolder(root=VAL_DIR, transform=get_val_transform())
test_dataset = ImageFolder(root=TEST_DIR, transform=get_val_transform())

print("Classes:", train_dataset.class_to_idx)
print(f"Train images: {len(train_dataset)}")
print(f"Val images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")

train_targets = train_dataset.targets
class_counts = np.bincount(train_targets)
print("Class counts:", class_counts)

class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
sample_weights = class_weights[train_targets]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

BATCH_SIZE = 32
NUM_WORKERS = 8

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

model = efficientnet_b4(weights="DEFAULT")
in_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, 2)
)


FREEZE_UNTIL_BLOCK = 3  # model.features has 9 blocks (indices 0-8); freeze 0-2
for idx, child in enumerate(model.features.children()):
    if idx < FREEZE_UNTIL_BLOCK:
        for p in child.parameters():
            p.requires_grad = False

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"[INFO] Trainable params: {trainable:,} / {total:,}")


class FocalLoss(nn.Module):
    def __init__(self, alpha_fake=0.15, alpha_real=0.85, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.class_alpha = torch.tensor([alpha_fake, alpha_real])
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        alpha_t = self.class_alpha.to(inputs.device)[targets]
        loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

criterion = FocalLoss(alpha_fake=0.15, alpha_real=0.85, gamma=2.0, label_smoothing=0.05)


backbone_params = [
    p for n, p in model.named_parameters()
    if p.requires_grad and not n.startswith("classifier")
]
head_params = [
    p for n, p in model.named_parameters()
    if p.requires_grad and n.startswith("classifier")
]

optimizer = optim.AdamW(
    [
        {"params": backbone_params, "lr": 5e-5},
        {"params": head_params, "lr": 5e-4},
    ],
    weight_decay=5e-4,
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6
)

scaler = GradScaler(enabled=torch.cuda.is_available())

START_EPOCH = 0
BEST_VAL_F1 = 0.0
BEST_VAL_ACC = 0.0
EPOCHS_NO_IMPROVE = 0

NUM_EPOCHS = 100

PATIENCE = 15
MAX_TRAIN_VAL_GAP = 15

checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoint_latest.pth")
best_model_path = os.path.join(PROJECT_ROOT, "best_efficientnet_b4.pth")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Keep the existing model instance so optimizer parameter groups remain valid.
    model.load_state_dict(checkpoint["model"].state_dict())
    model = model.to(device)



    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    START_EPOCH = checkpoint["epoch"] + 1
    BEST_VAL_F1 = checkpoint.get("best_val_f1", 0.0)
    BEST_VAL_ACC = checkpoint.get("best_val_acc", 0.0)
    EPOCHS_NO_IMPROVE = checkpoint.get("epochs_no_improve", 0)
    print(f"[INFO] Resuming weights from epoch {START_EPOCH} "
          f"(optimizer/scheduler reset since trainable params changed)")

    print(f"[INFO] Best val F1 loaded from checkpoint (bar to beat): {BEST_VAL_F1:.4f}")

    if START_EPOCH >= NUM_EPOCHS:
        print(f"[WARN] START_EPOCH ({START_EPOCH}) >= NUM_EPOCHS ({NUM_EPOCHS}); "
              f"no additional training will run. Raise NUM_EPOCHS if you intend to keep training.")

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, max_grad_norm=5.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_mixup = np.random.rand() < 0.3
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.25)
            with autocast("cuda"):
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of 'real'
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    epoch_f1 = f1_score(all_labels, all_preds, pos_label=1, average="binary")
    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_probs

print("Starting training...")

for epoch in range(START_EPOCH, NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
    val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
    scheduler.step()

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}"
    )


    gap = train_acc - val_acc
    if gap > MAX_TRAIN_VAL_GAP:
        print(f"[WARN] Train/Val accuracy gap is {gap:.2f}pp — model may be overfitting.")

    if val_f1 > BEST_VAL_F1:
        BEST_VAL_F1 = val_f1
        BEST_VAL_ACC = val_acc
        EPOCHS_NO_IMPROVE = 0
        torch.save(model, best_model_path)
        print(f"[OK] Best model saved. Validation F1: {val_f1:.4f}")
    else:
        EPOCHS_NO_IMPROVE += 1
        print(f"[INFO] No val F1 improvement for {EPOCHS_NO_IMPROVE}/{PATIENCE} epoch(s).")

    torch.save(
        {
            "epoch": epoch,
            "model": model,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_acc": BEST_VAL_ACC,
            "best_val_f1": BEST_VAL_F1,
            "epochs_no_improve": EPOCHS_NO_IMPROVE,
        },
        checkpoint_path
    )

    if EPOCHS_NO_IMPROVE >= PATIENCE:
        print(f"[INFO] Early stopping at epoch {epoch + 1}: no val F1 improvement for {PATIENCE} epochs.")
        break

best_model = torch.load(best_model_path, map_location=device, weights_only=False)
best_model.to(device)
best_model.eval()

test_loss, test_acc, test_f1, test_labels, test_probs = validate(best_model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test F1: {test_f1 * 100:.2f}%")

test_preds_default = [1 if p >= 0.5 else 0 for p in test_probs]
print("\nPer-class report (default 0.5 threshold):")
print(classification_report(test_labels, test_preds_default, target_names=["fake", "real"]))


_, _, _, val_labels, val_probs = validate(best_model, val_loader, criterion, device)
precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx = int(np.argmax(f1s[:-1]))  # last precision/recall pair has no threshold
best_threshold = thresholds[best_idx]
print(f"\n[INFO] Best threshold for 'real' F1 on VAL set: {best_threshold:.3f} "
      f"(val F1 at this threshold: {f1s[best_idx]:.4f})")

test_preds_tuned = [1 if p >= best_threshold else 0 for p in test_probs]
print(f"\nPer-class report (tuned threshold = {best_threshold:.3f}, applied to TEST):")
print(classification_report(test_labels, test_preds_tuned, target_names=["fake", "real"]))