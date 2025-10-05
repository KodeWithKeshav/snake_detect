from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model once at startup
MODEL_PATH = os.environ.get('MODEL_PATH', 'snake_species_classifier.h5')
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Snake species labels
class_labels = [
    'Common Krait',     
    'Banded Krait',     
    'Russell\'s Viper',         
    'Spectacled Cobra',             
    'King Cobra',     
    'Rat Snake',        
    'Bamboo Pit Viper', 
    'Malabar Pit Viper'
]

# [Keep your snake_info_db as is]
snake_info_db = {
    # ... (your existing snake info)
}

def prepare_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    temp_dir = "uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        file.save(file_path)
        img_array = prepare_image(file_path)
        predictions = model.predict(img_array)
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx] * 100)
        predicted_species = class_labels[pred_idx]

        # Get detailed information
        snake_details = snake_info_db.get(predicted_species, {})

        response_data = {
            "species": predicted_species,
            "confidence": confidence,
            "venomous": "Yes" if snake_details.get("venomous") else "No",
            "scientific_name": snake_details.get("scientific_name", "Unknown"),
            "venom_type": snake_details.get("venom_type", "N/A"),
            "danger_level": snake_details.get("danger_level", "Unknown"),
            "description": snake_details.get("description", ""),
            "venom_effects": snake_details.get("venom_effects", ""),
            "first_aid": snake_details.get("first_aid", []),
            "precautions": snake_details.get("precautions", ""),
            "habitat": snake_details.get("habitat", "")
        }

        logger.info(f"Prediction: {predicted_species} ({confidence:.2f}%)")
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host="0.0.0.0", port=port, debug=False)