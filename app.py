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

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate config keys
required_keys = ["image_size", "model_path", "min_confidence", "max_file_size_mb", "min_image_pixels"]
for key in required_keys:
    if key not in CONFIG:
        st.error(f"❌ Missing config key: '{key}'. Please check config.py")
        st.stop()

# Page config
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    .title { text-align: center; color: #2e7d32; font-size: 2.5em; font-weight: bold; }
    .subtitle { text-align: center; color: #555; font-size: 1.1em; margin-bottom: 30px; }
    .footer { text-align: center; color: #aaa; font-size: 0.8em; margin-top: 50px; }
    </style>
""", unsafe_allow_html=True)

# Load model
model = load_model_safe()

# Dynamic supported plants from CLASS_NAMES
supported_plants = sorted(set([name.split(' ')[0] for name in CLASS_NAMES if name != 'PlantVillage']))

# Header
st.markdown('<div class="title">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered plant health analysis using Deep Learning</div>', unsafe_allow_html=True)
st.divider()

# Sidebar
st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Minimum Confidence Threshold (%)",
    min_value=0, max_value=100,
    value=int(CONFIG["min_confidence"]),
    help="Predictions below this threshold will show a warning"
)
reject_low_confidence = st.sidebar.checkbox(
    "Reject predictions below threshold",
    value=False,
    help="If checked, predictions below threshold won't be shown"
)
st.sidebar.info(f"Current threshold: **{confidence_threshold}%**")
st.sidebar.divider()
st.sidebar.subheader("🌱 Supported Plants")
for plant in supported_plants:
    st.sidebar.write(f"✅ {plant}")

# How to Use
with st.expander("ℹ️ How to Use"):
    st.markdown(f"""
    <div style="background-color:#e8f5e9; padding:15px; border-radius:10px;">
        <b style="color:#2e7d32;">📌 Instructions:</b>
        <ol style="color:#333;">
            <li>Upload a clear image of a plant leaf (JPG/PNG)</li>
            <li>Make sure the leaf is clearly visible and well-lit</li>
            <li>Recommended image size: at least {CONFIG['image_size']}x{CONFIG['image_size']} pixels</li>
            <li>Supported plants: {', '.join(supported_plants)}</li>
            <li>The AI will detect disease and show treatment advice</li>
        </ol>
        <b style="color:#2e7d32;">✅ Supported Diseases:</b>
        <p style="color:#333;">Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus</p>
        <b style="color:#2e7d32;">📁 Max File Size:</b>
        <p style="color:#333;">{CONFIG['max_file_size_mb']}MB per image</p>
    </div>
    """, unsafe_allow_html=True)

# Multi-image upload
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

    for idx, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"🌿 Image {idx+1}: {uploaded_file.name}")

        # File size check
        if uploaded_file.size > CONFIG["max_file_size_mb"] * 1024 * 1024:
            st.error(f"❌ '{uploaded_file.name}' is too large! Max size is {CONFIG['max_file_size_mb']}MB.")
            continue

        image, img_array = preprocess_image(uploaded_file)
        if image is None:
            continue

        centered_image(image, f"{uploaded_file.name} ({image.size[0]}x{image.size[1]}px)")

        # Predict with timeout tracking
        with st.spinner(f"🔍 Analyzing {uploaded_file.name}..."):
            start_time = time.time()
            prediction = predict_safe(model, img_array)
            processing_time = time.time() - start_time

        if prediction is None:
            st.error("❌ Prediction failed! Please try again.")
            continue

        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)
        safe_class = html.escape(predicted_class)

        logger.info(f"[{uploaded_file.name}] {predicted_class} ({confidence:.2f}%) in {processing_time:.2f}s")

        st.divider()
        st.caption(f"⏱️ Analyzed in {processing_time:.2f}s")

        # Reject low confidence if enabled
        if reject_low_confidence and confidence < confidence_threshold:
            st.error(f"❌ Confidence too low ({confidence:.1f}%)! Please upload a clearer image.")
            continue

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌱 Detected", predicted_class)
        with col2:
            st.metric("🎯 Confidence", f"{confidence:.2f}%")

        st.write("Confidence Level:")
        st.progress(int(confidence))

        # Low confidence warning
        if confidence < confidence_threshold:
            st.warning(f"⚠️ Low confidence ({confidence:.1f}%)! Try a clearer image.")

        # Top 3 predictions
        with st.expander("🔢 Top 3 Predictions"):
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            for i, top_idx in enumerate(top_3_indices):
                name = CLASS_NAMES[top_idx]
                score = float(prediction[0][top_idx] * 100)
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{emoji} {name}")
                    st.progress(int(score))
                with col2:
                    st.write(f"**{score:.1f}%**")

        # Result
        is_healthy = "healthy" in predicted_class.lower()
        details = DISEASE_DATABASE.get(predicted_class, {
            'severity': 'Unknown', 'severity_color': '#888',
            'cause': 'Unknown', 'cure': 'Consult an expert.',
            'prevention': 'Monitor regularly.'
        })

        if is_healthy:
            st.markdown("""
                <div style="background-color:#e8f5e9; padding:20px; border-radius:10px; margin-top:20px;">
                    <h3 style="color:#2e7d32;">✅ Healthy Plant</h3>
                    <p style="color:#333;">Your plant looks healthy! Keep up the good care.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f"""
                <div style="background-color:#fff3e0; padding:20px; border-radius:10px; margin-top:20px;">
                    <h3 style="color:#e65100;">⚠️ Disease Detected</h3>
                    <p style="color:#333;"><b>Severity:</b> <span style="color:{details['severity_color']}; font-weight:bold;">🌡️ {details['severity']}</span></p>
                    <p style="color:#333;"><b>🔬 Cause:</b> {html.escape(details['cause'])}</p>
                    <p style="color:#333;"><b>💊 Treatment:</b> {html.escape(details['cure'])}</p>
                    <p style="color:#333;"><b>🛡️ Prevention:</b> {html.escape(details['prevention'])}</p>
                </div>
            """, unsafe_allow_html=True)

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

    # Total time
    total_time = time.time() - total_start
    if len(uploaded_files) > 1:
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