# app.py
import streamlit as st
import numpy as np
import time
import logging
import html
import io
import csv
import cv2
import plotly.express as px
import pandas as pd
from PIL import Image

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
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="wide")

st.markdown("""
    <style>
    /* Nature/Green Earthy Theme */
    .stApp {
        background: linear-gradient(180deg, #f1f8e9 0%, #e8f5e9 50%, #f9fbe7 100%);
    }
    .stApp * {
        color: #1b5e20 !important;
    }
    p, li, label, span {
        color: #333333 !important;
    }
    @media (max-width: 768px) {
        .title { font-size: 1.8em !important; }
        .subtitle { font-size: 0.9em !important; }
        .stButton > button { width: 100% !important; }
    }
    .title {
        text-align: center;
        font-size: 2.8em;
        font-weight: bold;
        background: linear-gradient(90deg, #1b5e20, #4caf50, #8bc34a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
        padding: 10px 0;
    }
    .subtitle {
        text-align: center;
        color: #558b2f;
        font-size: 1.1em;
        margin-bottom: 30px;
        font-style: italic;
    }
    .footer {
        text-align: center;
        color: #795548;
        font-size: 0.8em;
        margin-top: 50px;
        font-style: italic;
    }
    .stButton > button {
        border-radius: 20px;
        padding: 10px 25px;
        font-size: 1em;
        background: linear-gradient(90deg, #2e7d32, #4caf50) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(46,125,50,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1b5e20, #388e3c) !important;
        box-shadow: 0 6px 15px rgba(46,125,50,0.4);
    }
    .stMetric {
        background: rgba(255,255,255,0.7);
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
    }
    .stExpander {
        background: rgba(255,255,255,0.6) !important;
        border-radius: 10px !important;
        border: 1px solid #c8e6c9 !important;
    }
    div[data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.7);
        padding: 20px;
        border-radius: 15px;
        border: 2px dashed #4caf50;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.5);
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #2e7d32;
        font-weight: bold;
    }
    .stSidebar {
        background: linear-gradient(180deg, #e8f5e9, #f1f8e9) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── Model Loading ──────────────────────────────────────────────────────────────
model = load_model_safe()

# ── Session State ──────────────────────────────────────────────────────────────
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []
if 'total_scans' not in st.session_state:
    st.session_state.total_scans = 0

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
    green_pixels = np.sum((g > r) & (g > b) & (g > 80))
    total_pixels = img_array.shape[0] * img_array.shape[1]
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.20

# ── Disease Area Highlighter ───────────────────────────────────────────────────
def highlight_disease_area(image):
    """Highlight potential disease areas on leaf using OpenCV."""
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Detect brown/yellow spots (disease indicators)
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 255])
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])

    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(mask_brown, mask_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

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
    st.sidebar.divider()
    st.sidebar.subheader("📊 Session Stats")
    st.sidebar.metric("Total Scans", st.session_state.total_scans)
    if st.session_state.scan_history:
        diseases = [s['disease'] for s in st.session_state.scan_history if 'healthy' not in s['disease'].lower()]
        if diseases:
            most_common = max(set(diseases), key=diseases.count)
            st.sidebar.metric("Most Common", most_common.split(' ')[0])
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.scan_history = []
        st.session_state.total_scans = 0
        st.rerun()
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

def render_dashboard():
    if not st.session_state.scan_history:
        st.info("📊 No data yet! Go to Detect Disease tab and scan some images first!")
        st.markdown("""
            <div style="text-align:center; padding:40px; color:#aaa;">
                <h1>📊</h1>
                <p>Your statistics will appear here after scanning images</p>
            </div>
        """, unsafe_allow_html=True)
        return

    st.subheader("📊 Statistics Dashboard")
    history_df = pd.DataFrame(st.session_state.scan_history)

    col1, col2, col3, col4 = st.columns(4)
    total = len(history_df)
    healthy = len(history_df[history_df['disease'].str.contains('healthy', case=False)])
    diseased = total - healthy
    avg_confidence = history_df['confidence'].mean()

    with col1:
        st.metric("🔍 Total Scans", total)
    with col2:
        st.metric("✅ Healthy", healthy)
    with col3:
        st.metric("⚠️ Diseased", diseased)
    with col4:
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        disease_counts = history_df['disease'].value_counts().reset_index()
        disease_counts.columns = ['Disease', 'Count']
        fig_pie = px.pie(
            disease_counts,
            values='Count',
            names='Disease',
            title='🥧 Disease Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            history_df,
            x='disease',
            y='confidence',
            title='📊 Confidence by Detection',
            color='confidence',
            color_continuous_scale='Greens',
            labels={'disease': 'Disease', 'confidence': 'Confidence (%)'}
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("📋 Scan History Table"):
        st.dataframe(history_df, use_container_width=True)

def render_results(prediction, predicted_class, confidence, processing_time, confidence_threshold, reject_low_confidence, image=None):
    """Display analysis results for a single image."""
    st.divider()
    st.caption(f"⏱️ Analyzed in {processing_time:.2f}s")

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

    is_healthy = "healthy" in predicted_class.lower()

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
        color = details.get('severity_color', '#888')
        st.error("⚠️ **Disease Detected!**")
        st.markdown(f"""
            <div style="background-color:#fff3e0; padding:15px; border-radius:10px; margin-top:10px;">
                <p style="color:#333;"><b>🌡️ Severity:</b> <span style="color:{color}; font-weight:bold;">{details['severity']}</span></p>
                <p style="color:#333;"><b>🔬 Cause:</b> {details['cause']}</p>
                <p style="color:#333;"><b>💊 Treatment:</b> {details['cure']}</p>
                <p style="color:#333;"><b>🛡️ Prevention:</b> {details['prevention']}</p>
            </div>
        """, unsafe_allow_html=True)

        # Highlight disease area
        if image is not None:
           
           st.subheader("🗺️ Disease Area Detection")
           highlighted = highlight_disease_area(image)
           centered_image(highlighted, "🔴 Red boxes show potential disease areas")

    return True

# ── Main App ───────────────────────────────────────────────────────────────────
render_header()
confidence_threshold, reject_low_confidence = render_sidebar()
render_how_to_use()

# Tabs
tab1, tab2 = st.tabs(["🔍 Detect Disease", "📊 Dashboard"])

with tab1:
    uploaded_files = st.file_uploader(
        "📤 Upload leaf image(s)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one or more leaf images for batch analysis"
    )

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

        if len(uploaded_files) > 1:
            st.write(f"📊 Processing {len(uploaded_files)} images...")
            batch_progress = st.progress(0)

        for idx, uploaded_file in enumerate(uploaded_files):
            if len(uploaded_files) > 1:
                batch_progress.progress((idx + 1) / len(uploaded_files))

            safe_name = html.escape(uploaded_file.name)
            st.subheader(f"🌿 Image {idx+1}: {safe_name}")

            if uploaded_file.size > get_config("max_file_size_mb", 10) * 1024 * 1024:
                st.error(f"❌ '{safe_name}' is too large!")
                continue

            image, img_array = preprocess_image(uploaded_file)
            if image is None:
                continue

            centered_image(image, f"{safe_name} ({image.size[0]}x{image.size[1]}px)")

            # Plant validation
            if not is_plant_image(image):
                st.error("❌ This doesn't look like a plant leaf image! Please upload a proper leaf photo.")
                continue

            with st.spinner(f"🔍 Analyzing {safe_name}..."):
                start_time = time.time()
                try:
                    prediction = predict_safe(model, img_array)
                    processing_time = time.time() - start_time
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    st.error("❌ Failed to analyze image. Please try again.")
                    continue

            if prediction is None or prediction.shape != (1, len(CLASS_NAMES)):
                st.error("❌ Invalid prediction format!")
                continue

            pred_idx = int(np.argmax(prediction[0]))
            if pred_idx >= len(CLASS_NAMES):
                st.error("❌ Model returned invalid prediction index!")
                continue

            predicted_class = CLASS_NAMES[pred_idx]
            confidence = float(np.max(prediction[0]) * 100)

            success = render_results(
                prediction, predicted_class, confidence,
                processing_time, confidence_threshold, reject_low_confidence, image
            )

            if success:
                st.session_state.total_scans += 1
                st.session_state.scan_history.append({
                    'image': uploaded_file.name,
                    'disease': predicted_class,
                    'confidence': confidence,
                    'time': processing_time
                })

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

        if len(uploaded_files) > 1:
            total_time = time.time() - total_start
            st.success(f"✅ Analyzed {len(uploaded_files)} images in {total_time:.2f}s!")

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

with tab2:
    render_dashboard()

st.divider()
st.markdown('<div class="footer">Built with TensorFlow & Streamlit | Plant Disease Detection System</div>', unsafe_allow_html=True)