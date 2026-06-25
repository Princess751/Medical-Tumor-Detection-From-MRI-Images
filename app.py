import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io
import json

app = Flask(__name__)
CORS(app)

# Global model variable
model = None

def load_model():
    """Load the pre-trained tumor detection model"""
    global model
    model_path = os.getenv('MODEL_PATH', '/app/models/tumor_model.h5')
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using dummy model for demonstration")
            model = create_dummy_model()
    else:
        print(f"Model not found at {model_path}, using dummy model for demonstration")
        model = create_dummy_model()

def create_dummy_model():
    """Create a simple CNN model for demonstration"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image_data, target_size=(224, 224)):
    """Preprocess MRI image for model input"""
    try:
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing error: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'MRI Tumor Detection API',
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image data
        image_data = image_file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Extract results
        tumor_probability = float(predictions[0][1])
        no_tumor_probability = float(predictions[0][0])
        
        result = {
            'status': 'success',
            'predictions': {
                'has_tumor': {
                    'probability': tumor_probability,
                    'percentage': round(tumor_probability * 100, 2)
                },
                'no_tumor': {
                    'probability': no_tumor_probability,
                    'percentage': round(no_tumor_probability * 100, 2)
                }
            },
            'classification': 'tumor detected' if tumor_probability > 0.5 else 'no tumor detected',
            'confidence': round(max(tumor_probability, no_tumor_probability) * 100, 2)
        }
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/info', methods=['GET'])
def info():
    """API information endpoint"""
    return jsonify({
        'name': 'MRI Tumor Detection API',
        'version': '1.0.0',
        'description': 'Deep learning API for detecting tumors in MRI scans',
        'endpoints': {
            'POST /predict': 'Upload an MRI image and get tumor detection prediction',
            'GET /health': 'Health check endpoint',
            'GET /info': 'API information'
        },
        'usage': {
            'example': 'curl -X POST -F "image=@mri_scan.jpg" http://localhost:5000/predict'
        }
    }), 200

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Get configuration from environment
    debug_mode = os.getenv('FLASK_DEBUG', 'False') == 'True'
    port = int(os.getenv('PORT', 5000))
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
