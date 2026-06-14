# 🌿 Plant Disease Detection System

An AI-powered web application that detects plant diseases from leaf images using Deep Learning (CNN).

## 🚀 Features
- 📤 Upload single or multiple leaf images (batch processing)
- 🤖 AI detects disease with confidence percentage
- 🌡️ Shows severity level (Mild/Moderate/Severe)
- 💊 Provides treatment and prevention advice
- ⚙️ Adjustable confidence threshold via sidebar
- 📥 Export results as CSV
- 🛡️ Input validation and error handling
- ⏱️ Processing time display
- 🔢 Top 3 predictions shown
- Built with TensorFlow and Streamlit

## 🛠️ Tech Stack
- Python 3.10
- TensorFlow / Keras
- Streamlit
- NumPy, Pillow
- Modular structure (config.py, disease_db.py, utils.py)

## 📊 Dataset
- PlantVillage Dataset (54,000+ images, 16 classes)
- Supported plants: Pepper, Potato, Tomato

## 🦠 Supported Diseases
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus

## 📁 Project Structure
```text
plant-disease-detection/
├── app.py          ← Main Streamlit web app
├── config.py       ← Configuration settings
├── disease_db.py   ← Disease database
├── utils.py        ← Helper functions
├── model/
│   └── train.py    ← CNN model training
├── model.h5        ← Trained model
└── requirements.txt
```
## ▶️ How to Run
```bash
# Clone the repository
git clone https://github.com/srilakshmi21206/plant-disease-detection.git
cd plant-disease-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## ⚙️ Configuration
Edit `config.py` to customize:
- Image size
- Model path
- Confidence threshold
- Max file size


## 👩‍💻 Built By
Sri Lakshmi K — 

## 📄 License
This project is open source and available for educational purposes.
