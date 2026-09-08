import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Configuration & Paths
MODEL_PATH = "best_efficientnet_b4.pth"
PREVIOUS_MODEL_PATH = "best_efficientnet_b4_generalized.pth"
CHECKPOINT_PATH = "efficientnet_b4_training_checkpoint.pth"
FINAL_MODEL_PATH = "best_efficientnet_b4_generalized.pth"

SECOND_DATASET_ROOT = (
    "/DATA/deeksha/datasets/deep_dataset/Deep_Fake_Detection/TestData/real_vs_fake/real-vs-fake"
)
TRAIN_DIR = os.path.join(SECOND_DATASET_ROOT, "train")

SAMPLES_PER_CLASS_FOR_FINETUNE = 10000
TOTAL_EPOCHS = 100
PREVIOUS_COMPLETED_EPOCHS = 50  # Restored missing variable
BATCH_SIZE = 64
NUM_WORKERS = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 42

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Data Augmentations
train_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset_full = ImageFolder(TRAIN_DIR, transform=train_transform)
print(f"\nClass mapping: {train_dataset_full.class_to_idx}")
print(f"Total training images: {len(train_dataset_full)}")

# Class-Balanced Sampling
print("\nCreating balanced training subset...")
targets = np.array(train_dataset_full.targets)
rng = np.random.default_rng(SEED)

train_indices = []
for class_idx in np.unique(targets):
    class_indices = np.where(targets == class_idx)[0]
    n_samples = min(SAMPLES_PER_CLASS_FOR_FINETUNE, len(class_indices))
    selected_indices = rng.choice(class_indices, size=n_samples, replace=False)
    train_indices.extend(selected_indices.tolist())

random.shuffle(train_indices)
finetune_dataset = Subset(train_dataset_full, train_indices)
print(f"Training subset: {len(finetune_dataset)} images")

finetune_loader = DataLoader(
    finetune_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
print(f"Batches per epoch: {len(finetune_loader)}")

# Load Base Model Architecture
if os.path.exists(CHECKPOINT_PATH):
    print(f"\nTraining checkpoint found: {CHECKPOINT_PATH}")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    loading_mode = "checkpoint"
elif os.path.exists(PREVIOUS_MODEL_PATH):
    print(f"\nPrevious trained model found: {PREVIOUS_MODEL_PATH}")
    model = torch.load(PREVIOUS_MODEL_PATH, map_location=device, weights_only=False)
    loading_mode = "previous_model"
else:
    print(f"\nNo previous model found. Loading base model: {MODEL_PATH}")
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    loading_mode = "new_training"

model = model.to(device)

# Parameter Freezing Strategy
print("\nFreezing model parameters...")
for param in model.parameters():
    param.requires_grad = False

print("Unfreezing EfficientNet feature block 8...")
for name, child in model.features.named_children():
    if name == "8":
        for param in child.parameters():
            param.requires_grad = True

print("Unfreezing classifier...")
for param in model.classifier.parameters():
    param.requires_grad = True

# Parameter Counting
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print("\n" + "=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable ratio:      {100 * trainable_params / total_params:.2f}%")

# Optimization & Loss Setup
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)
scaler = GradScaler(device.type, enabled=torch.cuda.is_available())

start_epoch = 0
training_history = []

# Resolve Resume Logic
if loading_mode == "checkpoint":
    print("\n" + "=" * 70 + "\nRESUMING FROM CHECKPOINT\n" + "=" * 70)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1
    training_history = checkpoint.get("history", [])
    
    print(f"Last completed epoch: {checkpoint['epoch'] + 1}")
    print(f"Resuming at epoch:     {start_epoch + 1}/{TOTAL_EPOCHS}")

elif loading_mode == "previous_model":
    print("\n" + "=" * 70 + "\nRESUMING FROM 50-EPOCH MODEL\n" + "=" * 70)
    start_epoch = PREVIOUS_COMPLETED_EPOCHS
    print(f"Starting at epoch: {start_epoch + 1}/{TOTAL_EPOCHS}")

else:
    print("\n" + "=" * 70 + "\nSTARTING NEW TRAINING\n" + "=" * 70)
    start_epoch = 0

# Guard against execution if already trained
if start_epoch >= TOTAL_EPOCHS:
    print(f"\nTraining complete ({start_epoch}/{TOTAL_EPOCHS} epochs executed). Increase TOTAL_EPOCHS to continue.")
    raise SystemExit

# Training Loop
print("\n" + "=" * 70)
print(f"STARTING TRAINING: Epochs {start_epoch + 1} to {TOTAL_EPOCHS}")
print("=" * 70)

for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(finetune_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics Tracking
        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    # Epoch Summary
    epoch_loss = running_loss / len(finetune_loader)
    epoch_accuracy = 100.0 * correct / total_samples

    print(f"\nEpoch [{epoch + 1}/{TOTAL_EPOCHS}] | Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.2f}%")

    training_history.append({
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
    })

    # Save Checkpoint State
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": training_history,
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

print("\n" + "=" * 70 + "\nTRAINING COMPLETED\n" + "=" * 70)

# Save Final Target Artifact
torch.save(model, FINAL_MODEL_PATH)
print(f"Saved final model to: {FINAL_MODEL_PATH}")
print(f"Checkpoint retained at: {CHECKPOINT_PATH}\nDone!")