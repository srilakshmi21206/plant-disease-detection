# app.py
import streamlit as st
import numpy as np
import time
import logging
import html
import io
import csv

from config import CONFIG
from disease_db import CLASS_NAMES, DISEASE_DATABASE
from utils import load_model_safe, preprocess_image, predict_safe, centered_image

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_TOP_PREDICTIONS = 3
MIN_CONFIDENCE_DEFAULT = 70
COLUMN_RATIO = [3, 1]
MEDALS = ["🥇", "🥈", "🥉"]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config Validation ──────────────────────────────────────────────────────────
def get_config(key, default=None):
    return CONFIG.get(key, default)

required_keys = ["image_size", "model_path", "min_confidence", "max_file_size_mb", "min_image_pixels"]
for key in required_keys:
    if key not in CONFIG:
        st.error(f"❌ Missing config key: '{key}'. Please check config.py")
        st.stop()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    .title { text-align: center; color: #2e7d32; font-size: 2.5em; font-weight: bold; }
    .subtitle { text-align: center; color: #555; font-size: 1.1em; margin-bottom: 30px; }
    .footer { text-align: center; color: #aaa; font-size: 0.8em; margin-top: 50px; }
    </style>
""", unsafe_allow_html=True)

# ── Model Loading ──────────────────────────────────────────────────────────────
model = load_model_safe()

# ── Dynamic Supported Plants ───────────────────────────────────────────────────
supported_plants = sorted(set([
    name.split(' ')[0] for name in CLASS_NAMES
    if name != 'PlantVillage' and ' ' in name
]))

# ── Plant Image Validator ──────────────────────────────────────────────────────
def is_plant_image(image):
    """Check if image contains enough green tones to be a plant leaf."""
    img_array = np.array(image.convert("RGB"))
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    green_pixels = np.sum((g > r) & (g > b) & (g > 60))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.10  # at least 10% green

# ── UI Components ──────────────────────────────────────────────────────────────
def render_header():
    st.markdown('<div class="title">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered plant health analysis using Deep Learning</div>', unsafe_allow_html=True)
    st.divider()

def render_sidebar():
    st.sidebar.title("⚙️ Settings")
    threshold = st.sidebar.slider(
        "Minimum Confidence Threshold (%)",
        min_value=0, max_value=100,
        value=int(get_config("min_confidence", MIN_CONFIDENCE_DEFAULT)),
        help="Predictions below this threshold will show a warning"
    )
    reject = st.sidebar.checkbox(
        "Reject predictions below threshold",
        value=False,
        help="If checked, predictions below threshold won't be shown"
    )
    st.sidebar.info(f"Current threshold: **{threshold}%**")
    st.sidebar.divider()
    st.sidebar.subheader("🌱 Supported Plants")
    for plant in supported_plants:
        st.sidebar.write(f"✅ {plant}")
    return threshold, reject

def render_how_to_use():
    with st.expander("ℹ️ How to Use"):
        st.markdown(f"""
        <div style="background-color:#e8f5e9; padding:15px; border-radius:10px;">
            <b style="color:#2e7d32;">📌 Instructions:</b>
            <ol style="color:#333;">
                <li>Upload a clear image of a plant leaf (JPG/PNG)</li>
                <li>Make sure the leaf is clearly visible and well-lit</li>
                <li>Recommended image size: at least {get_config('image_size')}x{get_config('image_size')} pixels</li>
                <li>Supported plants: {', '.join(supported_plants)}</li>
                <li>The AI will detect disease and show treatment advice</li>
            </ol>
            <b style="color:#2e7d32;">✅ Supported Diseases:</b>
            <p style="color:#333;">Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus</p>
            <b style="color:#2e7d32;">📁 Max File Size:</b>
            <p style="color:#333;">{get_config('max_file_size_mb')}MB per image</p>
        </div>
        """, unsafe_allow_html=True)

def render_results(prediction, predicted_class, confidence, processing_time, confidence_threshold, reject_low_confidence):
    """Display analysis results for a single image."""

    st.divider()
    st.caption(f"⏱️ Analyzed in {processing_time:.2f}s")

    # Reject low confidence
    if reject_low_confidence and confidence < confidence_threshold:
        st.error(f"❌ Confidence too low ({confidence:.1f}%)! Please upload a clearer image.")
        return False

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌱 Detected", predicted_class)
    with col2:
        st.metric("🎯 Confidence", f"{confidence:.2f}%")

    st.write("Confidence Level:")
    st.progress(min(int(confidence), 100))

    if confidence < confidence_threshold:
        st.warning(f"⚠️ Low confidence ({confidence:.1f}%)! Try a clearer image.")

    # Top predictions
    with st.expander("🔢 Top Predictions"):
        top_k = min(MAX_TOP_PREDICTIONS, len(prediction[0]))
        top_indices = np.argsort(prediction[0])[-top_k:][::-1]
        for i, top_idx in enumerate(top_indices):
            name = CLASS_NAMES[top_idx]
            score = float(prediction[0][top_idx] * 100)
            emoji = MEDALS[i] if i < len(MEDALS) else "📍"
            col1, col2 = st.columns(COLUMN_RATIO)
            with col1:
                st.write(f"{emoji} {name}")
                st.progress(min(int(score), 100))
            with col2:
                st.write(f"**{score:.1f}%**")

    # Disease/Healthy result
    is_healthy = "healthy" in predicted_class.lower()

    # Warn if disease not in database
    if predicted_class not in DISEASE_DATABASE:
        st.warning(f"⚠️ '{predicted_class}' not in database. Using generic guidance.")

    details = DISEASE_DATABASE.get(predicted_class, {
        'severity': 'Unknown', 'severity_color': '#888',
        'cause': 'Unknown', 'cure': 'Consult an expert.',
        'prevention': 'Monitor regularly.'
    })

    if is_healthy:
        st.success("✅ **Healthy Plant** — Your plant looks healthy! Keep up the good care.")
        st.balloons()
    else:
        color = details.get('severity_color', '#888').lstrip('#').lower()
        st.error("⚠️ **Disease Detected!**")
        st.markdown(f"🌡️ **Severity:** :{color}[{details['severity']}]")
        st.markdown(f"🔬 **Cause:** {details['cause']}")
        st.markdown(f"💊 **Treatment:** {details['cure']}")
        st.markdown(f"🛡️ **Prevention:** {details['prevention']}")

    return True

# ── Main App ───────────────────────────────────────────────────────────────────
render_header()
confidence_threshold, reject_low_confidence = render_sidebar()
render_how_to_use()

# Upload
uploaded_files = st.file_uploader(
    "📤 Upload leaf image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload one or more leaf images for batch analysis"
)

# Empty state
if not uploaded_files:
    st.info("👆 Start by uploading a plant leaf image above!")
    st.markdown("""
        <div style="text-align:center; padding:40px; color:#aaa;">
            <h1>🌿</h1>
            <p>Upload a leaf image to detect plant diseases</p>
        </div>
    """, unsafe_allow_html=True)

else:
    results_data = []
    total_start = time.time()

    # Batch progress bar
    if len(uploaded_files) > 1:
        st.write(f"📊 Processing {len(uploaded_files)} images...")
        batch_progress = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):

        # Update batch progress
        if len(uploaded_files) > 1:
            batch_progress.progress((idx + 1) / len(uploaded_files))

        # Sanitize file name
        safe_name = html.escape(uploaded_file.name)
        st.subheader(f"🌿 Image {idx+1}: {safe_name}")

        # File size check
        if uploaded_file.size > get_config("max_file_size_mb", 10) * 1024 * 1024:
            st.error(f"❌ '{safe_name}' is too large! Max size is {get_config('max_file_size_mb')}MB.")
            continue

        image, img_array = preprocess_image(uploaded_file)
        if image is None:
            continue

        centered_image(image, f"{safe_name} ({image.size[0]}x{image.size[1]}px)")

        # ── Plant Image Validation ─────────────────────────────────────────────
        if not is_plant_image(image):
            st.error("❌ This doesn't look like a plant leaf image. Please upload a proper leaf photo.")
            continue

        # Predict
        with st.spinner(f"🔍 Analyzing {safe_name}..."):
            start_time = time.time()
            try:
                prediction = predict_safe(model, img_array)
                processing_time = time.time() - start_time
            except Exception as e:
                logger.error(f"Prediction error for {uploaded_file.name}: {e}")
                st.error(f"❌ Failed to analyze image. Please try again.")
                continue

        # Validate prediction shape
        if prediction is None or prediction.shape != (1, len(CLASS_NAMES)):
            st.error("❌ Invalid prediction format! Please try again.")
            continue

        # Validate prediction index
        pred_idx = int(np.argmax(prediction[0]))
        if pred_idx >= len(CLASS_NAMES):
            st.error("❌ Model returned invalid prediction index!")
            continue

        predicted_class = CLASS_NAMES[pred_idx]
        confidence = float(np.max(prediction[0]) * 100)

        logger.warning(f"[{uploaded_file.name}] {predicted_class} ({confidence:.2f}%) in {processing_time:.2f}s")

        # Render results
        success = render_results(
            prediction, predicted_class, confidence,
            processing_time, confidence_threshold, reject_low_confidence
        )

        if success:
            details = DISEASE_DATABASE.get(predicted_class, {
                'severity': 'Unknown', 'severity_color': '#888',
                'cause': 'Unknown', 'cure': 'Consult an expert.',
                'prevention': 'Monitor regularly.'
            })
            results_data.append({
                'Image': uploaded_file.name,
                'Detected Disease': predicted_class,
                'Confidence (%)': f"{confidence:.2f}",
                'Severity': details['severity'],
                'Cause': details['cause'],
                'Treatment': details['cure'],
                'Prevention': details['prevention'],
                'Processing Time (s)': f"{processing_time:.2f}"
            })

        st.divider()

    # Complete batch progress
    if len(uploaded_files) > 1:
        total_time = time.time() - total_start
        st.success(f"✅ Analyzed {len(uploaded_files)} images in {total_time:.2f}s!")

    # Export CSV
    if results_data:
        st.subheader("📥 Export Results")
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=results_data[0].keys())
        writer.writeheader()
        writer.writerows(results_data)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=output.getvalue(),
            file_name="plant_disease_results.csv",
            mime="text/csv"
        )

st.divider()
st.markdown('<div class="footer">Built with TensorFlow & Streamlit | Plant Disease Detection System</div>', unsafe_allow_html=True)