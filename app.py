from pathlib import Path
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import efficientnet_b4


PROJECT_ROOT = Path(__file__).resolve().parent

# Default production checkpoint path
MODEL_CHECKPOINT_PATH = PROJECT_ROOT / "Finetune_model.pth"

IMG_SIZE = 380
CLASS_NAMES = ["fake", "real"]  # ImageFolder default: {'fake': 0, 'real': 1}
REAL_THRESHOLD = 0.632  # Tuned decision threshold for 'real' class
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



def build_model_architecture(num_classes: int = 2) -> nn.Module:
    """Reconstructs the exact EfficientNet-B4 architecture used in training."""
    model = efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: Path) -> nn.Module:
    """Loads PyTorch model state handling state_dict structures or serialized objects."""
    if not checkpoint_path.exists():
        # Fallback to base model if fine-tuned checkpoint is missing
        fallback_path = PROJECT_ROOT / "best_efficientnet_b4.pth"
        if fallback_path.exists():
            checkpoint_path = fallback_path
        else:
            raise FileNotFoundError(f"Model checkpoint file missing: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        model = build_model_architecture(len(CLASS_NAMES))
        
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def get_inference_transform() -> transforms.Compose:
    """Cached transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


@torch.no_grad()
def predict(model: nn.Module, image: Image.Image, real_threshold: float):
    """Executes forward pass and evaluates prediction against the decision threshold."""
    transform = get_inference_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    real_idx = CLASS_NAMES.index("real")
    fake_idx = CLASS_NAMES.index("fake")
    
    real_prob = float(probs[real_idx])
    fake_prob = float(probs[fake_idx])

    pred_label = "real" if real_prob >= real_threshold else "fake"
    confidence = real_prob if pred_label == "real" else fake_prob

    return pred_label, confidence, real_prob, fake_prob



st.set_page_config(
    page_title="Deepfake Detector", 
    page_icon="🛡️", 
    layout="centered"
)

st.title(" Deepfake Image Detector")
st.markdown(
    "Upload a face image to classify whether it is **Real** or **Fake** using an "
    "EfficientNet-B4 deep learning architecture."
)

# Clean Informational Sidebar (No Settings Sliders/Selectors)
with st.sidebar:
    st.header("System Information")
    st.markdown("**Backbone Architecture:**\nEfficientNet-B4")
    st.markdown(f"**Target Device:**\n`{DEVICE.upper()}`")
    st.markdown(f"**Input Resolution:**\n`{IMG_SIZE} x {IMG_SIZE}`")
    st.divider()
    st.caption("Running optimized inference pipeline.")

# Main Interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("Invalid image format. Please upload a valid JPG, PNG, or WEBP file.")
        st.stop()

    st.image(image, caption="Uploaded Image Preview", use_container_width=True)

    # Run inference directly on upload
    try:
        with st.spinner("Loading classifier..."):
            model = load_model(MODEL_CHECKPOINT_PATH)

        with st.spinner("Analyzing facial features..."):
            label, confidence, real_prob, fake_prob = predict(model, image, REAL_THRESHOLD)

        st.subheader("Analysis Results")
        
        # Verdict Banner
        if label == "fake":
            st.error(f" **Prediction: FAKE** ({confidence * 100:.2f}% Confidence)")
        else:
            st.success(f" **Prediction: REAL** ({confidence * 100:.2f}% Confidence)")

        # Detailed Metrics Section
        st.markdown("### Class Probabilities")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Real Probability", value=f"{real_prob * 100:.2f}%")
            st.progress(real_prob)
        with col2:
            st.metric(label="Fake Probability", value=f"{fake_prob * 100:.2f}%")
            st.progress(fake_prob)

        # Simplified Model Details Section
        with st.expander(" Detection Details"):
            st.markdown(
                "This prediction is generated by inspecting facial feature manipulation "
                "patterns learned across deepfake benchmark datasets."
            )
            st.markdown(f"- **Primary Classification:** `{label.upper()}`")
            st.markdown(f"- **Assigned Confidence:** `{confidence * 100:.2f}%`")

    except FileNotFoundError as fnf_err:
        st.error(f"{fnf_err}")
    except Exception as e:
        st.error(f"An unexpected error occurred during inference: {e}")

else:
    st.info("Upload an image using the picker above to begin analysis.")

st.divider()
st.caption(
    "**Disclaimer:** This application is intended for research and portfolio demonstration purposes. "
    "Generative models evolve rapidly; performance may vary on highly compressed or edited images."
)
