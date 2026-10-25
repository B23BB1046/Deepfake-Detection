# Deepfake Image Detection

A computer-vision pipeline for distinguishing real and deepfake face images using **EfficientNet-B4** and PyTorch. This project examines class imbalance, threshold selection, and the gap between in-domain and cross-dataset performance.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white) ![Streamlit](https://img.shields.io/badge/Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## Highlights
- EfficientNet-B4 pretrained on ImageNet and fine-tuned for binary classification.
- Imbalance-aware training using per-class focal-loss weights and WeightedRandomSampler.
- Validation-based decision-threshold tuning rather than relying only on the default 0.5 threshold.
- Cross-dataset evaluation on a different fake-image source.
- Lightweight target-domain fine-tuning experiment.
- Streamlit app for image uploads and interactive predictions.

## Results
### FaceForensics++ test split
| Experiment | Accuracy | Real-class F1 | Fake-class F1 |
|---|---:|---:|---:|
| Initial model | 92.93% | 0.76 | 0.96 |
| Per-class focal-loss reweighting | 93.76% | 0.78 | 0.96 |
| Threshold tuning (0.632) | ~94% | 0.79 | 0.97 |

The training split contains 52,002 images, including 43,809 fake and 8,193 real images. Because the classes are imbalanced, aggregate accuracy alone is insufficient; real-class F1 is also tracked.

### Cross-dataset generalization
| Evaluation | Accuracy | F1 |
|---|---:|---:|
| FF++ model, zero-shot on 20,000 out-of-domain GAN images | 52.95% | 0.5256 |
| After fine-tuning on ~10,000 target-domain images for 15 epochs | 88.56% | 0.8853 |

These results illustrate that high in-domain performance does not guarantee generalization to unseen generation methods. Fine-tuning improves performance on the evaluated target domain, but does not establish universal deepfake detection.

## Approach
1. **Data preparation:** organize images into train, validation, and test splits.
2. **Model:** use an ImageNet-pretrained EfficientNet-B4 with a binary classification head.
3. **Imbalance handling:** combine per-class focal-loss weights with balanced mini-batch sampling.
4. **Optimization:** AdamW, differential learning rates, cosine scheduling, mixed precision, gradient clipping, and early stopping.
5. **Threshold selection:** tune the operating threshold using validation precision–recall behaviour.
6. **Generalization test:** evaluate on a separate GAN-generated image set, then run a limited fine-tuning experiment.

## Repository structure
| File | Purpose |
|---|---|
| main.py | Main training pipeline |
| processing_data.py | Data preparation and processing |
| Finetune.py | Target-domain fine-tuning experiment |
| app.py | Streamlit inference demo |
| config.yaml | Configuration |
| requirements.txt | Python dependencies |
| csv/ | Dataset metadata files |

## Run the demo
1. Clone this repository and enter its directory:
   `git clone https://github.com/B23BB1046/Deepfake-Detection.git`
2. Create and activate a virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Launch the app with `streamlit run app.py`.

The app expects compatible model checkpoints and configuration. Large datasets and trained weights may not be included in a fresh clone; check the application configuration and provide required files before running inference.

## Limitations
- Performance depends on the datasets and manipulation methods represented during training.
- The cross-dataset experiment shows a substantial domain shift.
- A model score is not proof that an image is authentic or manipulated.
- Predictions can be affected by compression, resizing, image quality, and unseen generation methods.

## Future work
- Train and evaluate on multiple manipulation and generation families.
- Add video-level inference with frame sampling and temporal aggregation.
- Report calibration, ROC-AUC, precision–recall curves, and error analysis.
- Add tests, pinned dependencies, and a small sample dataset for smoke testing.

## Responsible use
This project is for research and educational purposes. Do not use predictions as the sole basis for identity, legal, employment, or other high-impact decisions.