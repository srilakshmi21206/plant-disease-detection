# disease_db.py
CLASS_NAMES = [
    'Pepper Bell Bacterial Spot', 'Pepper Bell Healthy',
    'Potato Early Blight', 'Potato Late Blight', 'Potato Healthy',
    'Tomato Bacterial Spot', 'Tomato Early Blight', 'Tomato Late Blight',
    'Tomato Leaf Mold', 'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato Mosaic Virus',
    'Tomato Healthy', 'PlantVillage'
]

DISEASE_DATABASE = {
    'Pepper Bell Bacterial Spot': {
        'severity': 'Moderate', 'severity_color': '#ff9800',
        'cause': 'Caused by Xanthomonas bacteria',
        'cure': 'Apply copper-based fungicides. Remove infected leaves.',
        'prevention': 'Use disease-free seeds. Rotate crops annually.'
    },
    'Pepper Bell Healthy': {
        'severity': 'None', 'severity_color': '#4caf50',
        'cause': 'N/A', 'cure': 'N/A',
        'prevention': 'Keep up the good care!'
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
    'Potato Healthy': {
        'severity': 'None', 'severity_color': '#4caf50',
        'cause': 'N/A', 'cure': 'N/A',
        'prevention': 'Keep up the good care!'
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
    'Tomato Healthy': {
        'severity': 'None', 'severity_color': '#4caf50',
        'cause': 'N/A', 'cure': 'N/A',
        'prevention': 'Keep up the good care!'
    },
    'PlantVillage': {
        'severity': 'Unknown', 'severity_color': '#888',
        'cause': 'Unknown',
        'cure': 'Consult an agricultural expert.',
        'prevention': 'Monitor plant regularly.'
    },
}