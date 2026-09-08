# Deepfake image detection 

A deep learning pipeline for detecting deepfake face images, built on EfficientNet-B4 and trained on FaceForensics++ (FF++). Includes a full training pipeline with class-imbalance handling, a cross-dataset generalization experiment, and an interactive Streamlit demo app.

# Overview 

Deepfake detection models often perform well on the dataset they were trained on but fail to generalize to fakes produced by different generation methods. This project:
 
1. Trains a strong same-domain classifier on FF++
2. Diagnoses and fixes a real class-imbalance problem in the minority ("real") class
3. Honestly evaluates cross-dataset generalization and quantifies the drop
4. Demonstrates a lightweight fine-tuning strategy to partially recover generalization with a small amount of target-domain data
5. Ships an interactive app for live predictions

# Model & Approach 

- **Backbone:** EfficientNet-B4 (torchvision), pretrained on ImageNet
- **Classifier head:** `Dropout → Linear(512) → BatchNorm1d → ReLU → Dropout → Linear(2)`
- **Partial backbone freezing:** early blocks frozen, later blocks + head trainable — balances capacity against overfitting risk on a limited dataset
- **Loss:** Focal Loss with **per-class alpha weighting** (`alpha_fake=0.15`, `alpha_real=0.85`) to address a ~5.3:1 fake:real class imbalance, plus label smoothing
- **Sampling:** `WeightedRandomSampler` for balanced mini-batches, on top of the loss-level reweighting
- **Augmentation:** horizontal flip, mild rotation/affine/color jitter, random erasing, and mixup — tuned down from an initial, overly aggressive augmentation pass that was suppressing real training signal
- **Training:** AdamW with differential learning rates (backbone vs. head), cosine annealing with warm restarts, mixed-precision (AMP), gradient clipping, early stopping on validation F1
- **Decision threshold:** tuned via precision-recall analysis on the validation set (rather than a flat 0.5 cutoff) to better balance precision/recall on the minority class

# Dataset

**FaceForensics++ (FF++)**
 
| Split | Images |
|---|---|
| Train | 52,002 |
| Val | 8,090 |
| Test | 15,240 |
 
Class distribution (train): 43,809 fake / 8,193 real (~5.3:1 imbalance).

## Results
 
### Same-domain (FF++ test set)
 
| Stage | Test Accuracy | Real F1 | Fake F1 |
|---|---|---|---|
| Initial model | 92.93% | 0.76 | 0.96 |
| + per-class focal loss reweighting | 93.76% | 0.78 | 0.96 |
| + tuned decision threshold (0.632) | ~94% | 0.79 | 0.97 |
 
The key challenge was **minority-class ("real") performance**, not overall accuracy — the initial model's aggregate accuracy looked reasonable while real-class precision/recall lagged well behind fake. Addressing this required per-class loss reweighting (a global focal-loss `alpha` scalar does *not* reweight by class — this was corrected to an indexed per-class weight) and post-hoc threshold tuning, both of which measurably improved real-class detection without hurting fake-class performance.


### Cross-dataset generalization
 
Evaluating the FF++-trained model on a held-out set from a different deepfake dataset (GAN-generated faces, 20,000 images) revealed a substantial generalization gap:
 
| | Accuracy | F1 |
|---|---|---|
| Zero-shot (FF++ model, no adaptation) | 52.95% | 0.5256 |
 
Near chance-level performance — a known, documented limitation in deepfake detection: models trained on one manipulation method (FF++'s face-swap/reenactment videos) don't reliably transfer to a different generation method (GAN-synthesized stills) without adaptation, because the model learns artifacts specific to the training pipeline rather than fully general "fakeness."

### Lightweight cross-dataset fine-tuning
 
Rather than fully retraining on the second dataset, a small, targeted fine-tune was applied:
 
- ~10,000 images sampled from the second dataset's training split (a small fraction of the full dataset)
- Only the final backbone block + classifier head unfrozen (to adapt without catastrophically forgetting FF++ features)
- Low learning rate (`1e-5`), 15 epochs
- A held-out slice monitored throughout to confirm the model wasn't just memorizing the adaptation set
| | Accuracy | F1 |
|---|---|---|
| Before fine-tuning | 52.95% | 0.5256 |
| After lightweight fine-tuning | 88.56% | 0.8853 |
 
+10 points of accuracy from a small, targeted adaptation, with train/holdout accuracy tracking closely throughout training (no overfitting to the small fine-tune slice).

## Project Structure
 
.
├── main.py                      
├── processing_data.py                  
├── Finetune.py           
├── app.py                       
├── csv/
│   └── FF++_Metadata.csv         
├── data/
│   ├── train/{real,fake}/
│   ├── val/{real,fake}/
│   └── test/{real,fake}/
├── best_efficientnet_b4.pth      
├── Finetune_model.pth 
└── README.md 
|__ requirements.txt


**Launch the demo app:**

streamlit run app.py

Lets you upload an image, choose between the FF++ and generalized checkpoints, and adjust the decision threshold interactively.

## Key Engineering Challenges Addressed
 
- **Class imbalance in the loss function:** the initial focal loss implementation used a single global `alpha` scalar, which does not perform per-class reweighting despite appearing to. Fixed by indexing `alpha` per class.
- **Checkpoint/optimizer compatibility:** changing which backbone layers were frozen invalidated the saved optimizer state (`param_groups` size mismatch on `load_state_dict`); resolved by reinitializing the optimizer/scheduler fresh whenever trainable parameters change, while still resuming model weights.
- **Off-by-one epoch resuming:** `NUM_EPOCHS <= START_EPOCH` after a checkpoint resume silently skipped training entirely (empty `range()`); added an explicit warning to catch this in future runs.
- **Decision threshold vs. `argmax`:** the default 0.5 threshold implicit in `argmax` is not optimal for an imbalanced task; post-hoc threshold tuning via `precision_recall_curve` on the validation set gave a free (no retraining) improvement.


## Limitations & Future Work
 
- Performance is strongest on manipulation methods similar to FF++; cross-dataset generalization, while improved by lightweight fine-tuning, still lags same-domain performance.
- Further generalization gains would likely require training on a mix of multiple deepfake datasets (varied generation methods) rather than adapting after the fact.
- Frequency-domain or dual-stream input representations (a known technique in the deepfake-detection literature) are a promising direction not explored here.

