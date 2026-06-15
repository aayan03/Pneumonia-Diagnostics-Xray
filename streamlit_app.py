"""Streamlit web application for pneumonia detection from chest X-rays."""

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH, create_required_directories
from gradcam import get_gradcam_image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed",
)

create_required_directories()

# ---------------------------------------------------------------------------
# Premium CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Animated gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 70%, #0d1b2a 100%);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 860px; }

/* ── Hero header ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    animation: fadeInDown 0.8s ease;
}
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to   { opacity: 1; transform: translateY(0); }
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00d4aa22, #0ea5e922);
    border: 1px solid #00d4aa44;
    color: #00d4aa;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #00d4aa 50%, #0ea5e9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.75rem;
}
.hero-subtitle {
    color: #8b9ab5;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Stat cards row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
    animation: fadeIn 1s ease 0.3s both;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.stat-card {
    flex: 1;
    background: linear-gradient(135deg, #131d2e, #0d1828);
    border: 1px solid #1e3a5f44;
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-3px);
    border-color: #00d4aa55;
}
.stat-value {
    font-family: 'Sora', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00d4aa;
}
.stat-label {
    font-size: 0.75rem;
    color: #5a7a9a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* ── Upload section ── */
.upload-wrapper {
    background: linear-gradient(135deg, #131d2e, #0d1828);
    border: 2px dashed #1e3a5f;
    border-radius: 18px;
    padding: 2rem 1.5rem;
    margin: 1.5rem 0;
    text-align: center;
    transition: border-color 0.3s ease;
    animation: fadeIn 1s ease 0.5s both;
}
.upload-wrapper:hover { border-color: #00d4aa66; }
.upload-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.upload-title {
    font-family: 'Sora', sans-serif;
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.upload-hint { color: #5a7a9a; font-size: 0.85rem; }

/* ── Streamlit file uploader overrides ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #0d1828 !important;
    border: 1.5px dashed #1e3a5f !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #00d4aa !important;
    background: #0f1f35 !important;
}

/* ── Result card ── */
.result-card {
    border-radius: 18px;
    padding: 1.8rem;
    margin: 1.5rem 0;
    animation: slideUp 0.6s ease;
    border: 1px solid transparent;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-normal {
    background: linear-gradient(135deg, #062318, #0a2e1c);
    border-color: #00c85133;
}
.result-pneumonia {
    background: linear-gradient(135deg, #2a0a0a, #1e1010);
    border-color: #ff444433;
}
.result-label {
    font-family: 'Sora', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-normal .result-label   { color: #00e676; }
.result-pneumonia .result-label { color: #ff5252; }
.result-confidence {
    color: #8b9ab5;
    font-size: 0.95rem;
    margin-bottom: 1.2rem;
}

/* ── Probability bar ── */
.prob-bar-bg {
    background: #1a2a3a;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.8rem;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    animation: growBar 1.2s cubic-bezier(0.4,0,0.2,1);
}
@keyframes growBar {
    from { width: 0%; }
}
.prob-bar-normal   { background: linear-gradient(90deg, #00b894, #00e676); }
.prob-bar-pneumonia { background: linear-gradient(90deg, #d63031, #ff5252); }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #5a7a9a;
    margin-bottom: 0.3rem;
}

/* ── Image cards ── */
.img-card {
    background: linear-gradient(135deg, #131d2e, #0d1828);
    border: 1px solid #1e3a5f44;
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
    animation: fadeIn 0.8s ease;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.img-card:hover {
    transform: translateY(-2px);
    border-color: #00d4aa44;
}
.img-card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5a7a9a;
    margin-bottom: 0.6rem;
}

/* ── Section headings ── */
.section-heading {
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #cbd5e1;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-heading::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}

/* ── Disclaimer ── */
.disclaimer {
    background: #0d1828;
    border: 1px solid #1e3a5f44;
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    color: #4a6a8a;
    font-size: 0.8rem;
    text-align: center;
    margin-top: 1.5rem;
    line-height: 1.5;
}

/* ── Spinner override ── */
.stSpinner > div { border-top-color: #00d4aa !important; }

/* ── Divider ── */
hr { border-color: #1e3a5f33 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TensorFlow check
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf  # noqa: F401
except ImportError:
    st.error(
        "**TensorFlow is not installed.**\n\n"
        "Run: `pip install -r requirements.txt`"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    import tensorflow as tf
    if not MODEL_PATH.exists():
        st.error(
            "**Model file not found.**\n\n"
            "Train the model first: `python train.py`\n\n"
            f"Expected: `{MODEL_PATH}`"
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(pil_image):
    img = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">AI-Powered Medical Imaging</div>
    <div class="hero-title">PneumoScan AI</div>
    <div class="hero-subtitle">
        Upload a chest X-ray and receive an instant AI diagnosis with
        Grad-CAM visual explanations — in seconds.
    </div>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">EfficientNet</div>
        <div class="stat-label">Model Architecture</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">224×224</div>
        <div class="stat-label">Input Resolution</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">Grad-CAM</div>
        <div class="stat-label">Explainability</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">2-Class</div>
        <div class="stat-label">Normal / Pneumonia</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Upload section
# ---------------------------------------------------------------------------
st.markdown('<p class="section-heading">📂 Upload X-Ray Image</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your chest X-ray here",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; color:#3a5a7a; font-size:0.85rem; padding:0.5rem 0 1rem;">
        Supports PNG, JPG, JPEG &nbsp;|&nbsp; Max 200 MB
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inference + results
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")

    model = load_model()
    image_array = prepare_image(pil_image)

    with st.spinner("Analysing image with AI..."):
        pneumonia_prob = float(model.predict(image_array, verbose=0)[0][0])

    predicted_index = 1 if pneumonia_prob >= 0.5 else 0
    prediction = CLASS_NAMES[predicted_index]
    confidence = pneumonia_prob if predicted_index == 1 else 1.0 - pneumonia_prob
    conf_pct = confidence * 100
    pneumonia_pct = pneumonia_prob * 100
    normal_pct = (1 - pneumonia_prob) * 100

    # Result card
    if predicted_index == 1:
        icon = "⚠️"
        card_cls = "result-pneumonia"
    else:
        icon = "✅"
        card_cls = "result-normal"

    bar_cls = "prob-bar-pneumonia" if predicted_index == 1 else "prob-bar-normal"

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div class="result-label">{icon}&nbsp; {prediction}</div>
        <div class="result-confidence">Confidence: <strong>{conf_pct:.1f}%</strong></div>

        <div class="prob-label">
            <span>Normal</span><span>{normal_pct:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill prob-bar-normal" style="width:{normal_pct:.1f}%"></div>
        </div>

        <div class="prob-label" style="margin-top:0.6rem;">
            <span>Pneumonia</span><span>{pneumonia_pct:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill prob-bar-pneumonia" style="width:{pneumonia_pct:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grad-CAM
    st.markdown('<p class="section-heading">🔬 Visual Explanation (Grad-CAM)</p>', unsafe_allow_html=True)

    with st.spinner("Generating heatmap..."):
        suffix = Path(uploaded_file.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        try:
            gradcam_pil = get_gradcam_image(tmp_path, model)
        finally:
            tmp_path.unlink(missing_ok=True)

    col_orig, col_cam = st.columns(2)
    with col_orig:
        st.markdown('<div class="img-card"><div class="img-card-title">Original X-Ray</div>', unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_cam:
        st.markdown('<div class="img-card"><div class="img-card-title">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
        st.image(gradcam_pil, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ &nbsp;This tool is for <strong>research and educational purposes only</strong>.
        It is not a substitute for professional medical diagnosis.
        Always consult a qualified physician.
    </div>
    """, unsafe_allow_html=True)
