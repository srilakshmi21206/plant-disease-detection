import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
IMAGE_SIZE = 128
MODEL_PATH = 'model.h5'
CONFIDENCE_THRESHOLD = 70.0

# Page config
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f7f0; }
    .title { text-align: center; color: #2e7d32; font-size: 2.5em; font-weight: bold; }
    .subtitle { text-align: center; color: #555; font-size: 1.1em; margin-bottom: 30px; }
    .footer { text-align: center; color: #aaa; font-size: 0.8em; margin-top: 50px; }
    .how-to { background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
    'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus',
    'Tomato Healthy', 'PlantVillage'
]

DISEASE_DETAILS = {
    'Pepper Bell Bacterial Spot': {
        'severity': 'Moderate', 'severity_color': '#ff9800',
        'cause': 'Caused by Xanthomonas bacteria',
        'cure': 'Apply copper-based fungicides. Remove infected leaves. Avoid overhead watering.',
        'prevention': 'Use disease-free seeds. Rotate crops annually.'
    },
    'Potato Early Blight': {
        'severity': 'Mild', 'severity_color': '#4caf50',
        'cause': 'Caused by Alternaria solani fungus',
        'cure': 'Apply fungicides like chlorothalonil. Remove affected leaves.',
        'prevention': 'Avoid overhead irrigation. Ensure proper spacing.'
    },
    'Potato Late Blight': {
        'severity': 'Severe', 'severity_color': '#f44336',
        'cause': 'Caused by Phytophthora infestans',
        'cure': 'Apply metalaxyl fungicides immediately. Destroy infected plants.',
        'prevention': 'Use resistant varieties. Avoid wet conditions.'
    },
    'Tomato Bacterial Spot': {
        'severity': 'Moderate', 'severity_color': '#ff9800',
        'cause': 'Caused by Xanthomonas bacteria',
        'cure': 'Copper-based sprays. Remove infected parts.',
        'prevention': 'Use certified disease-free seeds.'
    },
    'Tomato Early Blight': {
        'severity': 'Mild', 'severity_color': '#4caf50',
        'cause': 'Caused by Alternaria solani',
        'cure': 'Apply mancozeb or chlorothalonil fungicide.',
        'prevention': 'Mulch around plants. Avoid wetting leaves.'
    },
    'Tomato Late Blight': {
        'severity': 'Severe', 'severity_color': '#f44336',
        'cause': 'Caused by Phytophthora infestans',
        'cure': 'Apply fungicides immediately. Remove all infected plants.',
        'prevention': 'Plant resistant varieties. Improve air circulation.'
    },
    'Tomato Leaf Mold': {
        'severity': 'Moderate', 'severity_color': '#ff9800',
        'cause': 'Caused by Passalora fulva fungus',
        'cure': 'Apply fungicides. Improve ventilation.',
        'prevention': 'Reduce humidity. Avoid overcrowding plants.'
    },
    'Tomato Septoria Leaf Spot': {
        'severity': 'Moderate', 'severity_color': '#ff9800',
        'cause': 'Caused by Septoria lycopersici',
        'cure': 'Remove infected leaves. Apply copper fungicide.',
        'prevention': 'Avoid overhead watering. Rotate crops.'
    },
    'Tomato Spider Mites': {
        'severity': 'Mild', 'severity_color': '#4caf50',
        'cause': 'Caused by Tetranychus urticae mites',
        'cure': 'Apply miticides or neem oil spray.',
        'prevention': 'Keep plants well watered. Avoid dusty conditions.'
    },
    'Tomato Target Spot': {
        'severity': 'Moderate', 'severity_color': '#ff9800',
        'cause': 'Caused by Corynespora cassiicola',
        'cure': 'Apply fungicides. Remove infected leaves.',
        'prevention': 'Improve air circulation. Avoid leaf wetness.'
    },
    'Tomato Yellow Leaf Curl Virus': {
        'severity': 'Severe', 'severity_color': '#f44336',
        'cause': 'Caused by Tomato yellow leaf curl virus (TYLCV)',
        'cure': 'No cure. Remove and destroy infected plants immediately.',
        'prevention': 'Control whitefly population. Use resistant varieties.'
    },
    'Tomato Mosaic Virus': {
        'severity': 'Severe', 'severity_color': '#f44336',
        'cause': 'Caused by Tomato mosaic virus (ToMV)',
        'cure': 'No cure. Remove infected plants to prevent spread.',
        'prevention': 'Disinfect tools. Wash hands before handling plants.'
    },
}

# ── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'model.h5' exists.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()

# ── Image Validation ───────────────────────────────────────────────────────────
def validate_and_preprocess(uploaded_file):
    try:
        image = Image.open(uploaded_file)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            st.info("ℹ️ Image converted to RGB format.")

        # Check image size
        width, height = image.size
        if width < 32 or height < 32:
            st.error("❌ Image too small! Please upload an image at least 32x32 pixels.")
            return None, None

        # Preprocess
        img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img_resized) / 255.0

        # Check channels
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            st.error("❌ Invalid image format. Please upload a color image.")
            return None, None

        img_array = np.expand_dims(img_array, axis=0)
        return image, img_array

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="title">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered plant health analysis using Deep Learning</div>', unsafe_allow_html=True)
st.divider()

# ── How to Use ─────────────────────────────────────────────────────────────────
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    <div style="background-color:#e8f5e9; padding:15px; border-radius:10px;">
        <b style="color:#2e7d32;">📌 Instructions:</b>
        <ol style="color:#333;">
            <li>Upload a clear image of a plant leaf (JPG/PNG)</li>
            <li>Make sure the leaf is clearly visible and well-lit</li>
            <li>Recommended image size: at least 128x128 pixels</li>
            <li>Supported plants: Pepper, Potato, Tomato</li>
            <li>The AI will detect disease and show treatment advice</li>
        </ol>
        <b style="color:#2e7d32;">✅ Supported Diseases:</b>
        <p style="color:#333;">Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus</p>
    </div>
    """, unsafe_allow_html=True)

# ── File Upload ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "📤 Upload a leaf image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a plant leaf. Max size: 200MB"
    )

if uploaded_file is not None:

    # Validate file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("❌ File too large! Please upload an image smaller than 10MB.")
        st.stop()

    image, img_array = validate_and_preprocess(uploaded_file)

    if image is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption=f"Uploaded Image ({image.size[0]}x{image.size[1]}px)", use_column_width=True)

        # Predict with timing
        with st.spinner("🔍 Analyzing leaf..."):
            start_time = time.time()
            prediction = model.predict(img_array, verbose=0)
            end_time = time.time()
            processing_time = end_time - start_time

        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        logger.info(f"Predicted: {predicted_class} with {confidence:.2f}% confidence")

        st.divider()
        st.subheader("📊 Analysis Results")

        # Processing time
        st.caption(f"⏱️ Analysis completed in {processing_time:.2f} seconds")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌱 Detected", predicted_class)
        with col2:
            st.metric("🎯 Confidence", f"{confidence:.2f}%")

        st.write("Confidence Level:")
        st.progress(int(confidence))

        # Low confidence warning
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f"⚠️ Low confidence ({confidence:.1f}%). Please upload a clearer image for better results.")

        # ── Top 3 Predictions ──────────────────────────────────────────────────
        st.subheader("🔢 Top 3 Predictions")
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        for i, idx in enumerate(top_3_indices):
            name = CLASS_NAMES[idx]
            score = prediction[0][idx] * 100
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{emoji} {name}")
                st.progress(int(score))
            with col2:
                st.write(f"**{score:.1f}%**")

        # ── Disease/Healthy Result ─────────────────────────────────────────────
        is_healthy = "healthy" in predicted_class.lower()

        if is_healthy:
            st.markdown("""
                <div style="background-color:#e8f5e9; padding:20px; border-radius:10px; margin-top:20px;">
                    <h3 style="color:#2e7d32;">✅ Healthy Plant</h3>
                    <p style="color:#333;">Your plant looks healthy! Keep up the good care.</p>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            details = DISEASE_DETAILS.get(predicted_class, {
                'severity': 'Unknown', 'severity_color': '#888',
                'cause': 'Unknown cause',
                'cure': 'Consult an agricultural expert.',
                'prevention': 'Monitor plant regularly.'
            })

            st.markdown(f"""
                <div style="background-color:#fff3e0; padding:20px; border-radius:10px; margin-top:20px;">
                    <h3 style="color:#e65100;">⚠️ Disease Detected</h3>
                    <p style="color:#333;"><b>Severity:</b> <span style="color:{details['severity_color']}; font-weight:bold;">🌡️ {details['severity']}</span></p>
                    <p style="color:#333;"><b>🔬 Cause:</b> {details['cause']}</p>
                    <p style="color:#333;"><b>💊 Treatment:</b> {details['cure']}</p>
                    <p style="color:#333;"><b>🛡️ Prevention:</b> {details['prevention']}</p>
                </div>
            """, unsafe_allow_html=True)

st.divider()
st.markdown('<div class="footer">Built with TensorFlow & Streamlit | Plant Disease Detection System</div>', unsafe_allow_html=True)