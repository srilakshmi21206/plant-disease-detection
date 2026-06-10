import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f7f0; }
    .title { text-align: center; color: #2e7d32; font-size: 2.5em; font-weight: bold; }
    .subtitle { text-align: center; color: #555; font-size: 1.1em; margin-bottom: 30px; }
    .footer { text-align: center; color: #aaa; font-size: 0.8em; margin-top: 50px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

model = load_model()

class_names = [
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
    'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus',
    'Tomato Healthy', 'PlantVillage'
]

disease_details = {
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

# Header
st.markdown('<div class="title">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered plant health analysis using Deep Learning</div>', unsafe_allow_html=True)
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Analyzing leaf..."):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.divider()
    st.subheader("📊 Analysis Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌱 Detected", predicted_class)
    with col2:
        st.metric("🎯 Confidence", f"{confidence:.2f}%")

    st.write("Confidence Level:")
    st.progress(int(confidence))

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
        details = disease_details.get(predicted_class, {
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