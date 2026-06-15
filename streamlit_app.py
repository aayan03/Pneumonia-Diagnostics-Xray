"""Streamlit web application for pneumonia detection from chest X-rays."""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH, create_required_directories
from gradcam import get_gradcam_image

st.set_page_config(page_title="PneumoScan AI", page_icon="🫁", layout="centered")

try:
    create_required_directories()
except Exception:
    pass  # non-fatal on read-only filesystems


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        import tensorflow as tf
    except ImportError:
        st.error("TensorFlow not installed. Run: `pip install -r requirements.txt`")
        st.stop()
    if not MODEL_PATH.exists():
        st.error(f"Model not found: `{MODEL_PATH}`. Run `python train.py` first.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(pil_image):
    img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🫁 PneumoScan AI")
st.caption("AI-Powered Pneumonia Detection from Chest X-Rays")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "EfficientNetB0")
col2.metric("Input", "224×224")
col3.metric("XAI", "Grad-CAM")
col4.metric("Classes", "2")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    model = load_model()

    with st.spinner("Analysing image..."):
        image_array = prepare_image(pil_image)
        pneumonia_prob = float(model.predict(image_array, verbose=0)[0][0])

    predicted_index = 1 if pneumonia_prob >= 0.5 else 0
    prediction = CLASS_NAMES[predicted_index]
    confidence  = pneumonia_prob if predicted_index == 1 else 1.0 - pneumonia_prob
    norm_pct  = (1 - pneumonia_prob) * 100
    pneu_pct  = pneumonia_prob * 100

    st.divider()

    if predicted_index == 1:
        st.error(f"### ⚠️ PNEUMONIA DETECTED — {confidence*100:.1f}% confidence")
    else:
        st.success(f"### ✅ NORMAL — {confidence*100:.1f}% confidence")

    c1, c2 = st.columns(2)
    c1.metric("Normal probability",    f"{norm_pct:.1f}%")
    c2.metric("Pneumonia probability", f"{pneu_pct:.1f}%")

    st.progress(pneu_pct / 100, text=f"Pneumonia risk: {pneu_pct:.1f}%")

    st.divider()
    st.subheader("🔬 Grad-CAM Heatmap")

    with st.spinner("Generating heatmap..."):
        suffix = Path(uploaded_file.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        try:
            gradcam_pil = get_gradcam_image(tmp_path, model)
            gradcam_ok  = True
        except Exception as e:
            gradcam_ok  = False
            gradcam_err = str(e)
        finally:
            tmp_path.unlink(missing_ok=True)

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.caption("Original X-Ray")
        st.image(pil_image, use_container_width=True)
    with img_col2:
        st.caption("Grad-CAM Heatmap")
        if gradcam_ok:
            st.image(gradcam_pil, use_container_width=True)
        else:
            st.warning(f"Heatmap unavailable: {gradcam_err}")

    st.divider()
    st.caption(
        "⚠️ For research and educational purposes only. "
        "Not a substitute for professional medical diagnosis."
    )
