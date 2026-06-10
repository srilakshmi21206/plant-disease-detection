# utils.py
import numpy as np
import tensorflow as tf
import streamlit as st
import logging
from PIL import Image
from config import CONFIG

logger = logging.getLogger(__name__)

@st.cache_resource
def load_model_safe():
    try:
        model = tf.keras.models.load_model(CONFIG["model_path"])
        logger.info("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'model.h5' exists in the project folder.")
        st.info("💡 Tip: Run 'python model/train.py' to generate the model file.")
        st.stop()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def preprocess_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)

        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            st.info("ℹ️ Image converted to RGB format.")

        # Validate size
        width, height = image.size
        min_px = CONFIG["min_image_pixels"]
        if width < min_px or height < min_px:
            st.error(f"❌ Image too small! Minimum size is {min_px}x{min_px} pixels.")
            return None, None

        # Resize and normalize
        size = CONFIG["image_size"]
        img_resized = image.resize((size, size))
        img_array = np.array(img_resized) / 255.0

        # Validate channels
        if len(img_array.shape) != 3 or img_array.shape[2] != 3:
            st.error("❌ Invalid image format. Please upload a color image.")
            return None, None

        return image, np.expand_dims(img_array, axis=0)

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None, None

def predict_safe(model, img_array):
    try:
        prediction = model.predict(img_array, verbose=0)
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        st.error(f"❌ Prediction failed: {str(e)}")
        return None

def centered_image(image, caption):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption=caption, use_column_width=True)