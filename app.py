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

# Model loading with compatibility patch
MODEL_PATH = os.environ.get('MODEL_PATH', 'snake_species_classifier.h5')
model = None

def patch_tensorflow_compatibility():
    """Patch TensorFlow to handle batch_shape parameter from newer versions"""
    try:
        # Store original InputLayer init
        original_input_layer_init = tf.keras.layers.InputLayer.__init__
        
        def patched_input_layer_init(self, input_shape=None, batch_size=None, 
                                     dtype=None, input_tensor=None, sparse=None,
                                     ragged=None, name=None, **kwargs):
            # Handle batch_shape from newer TF versions
            if 'batch_shape' in kwargs:
                batch_shape = kwargs.pop('batch_shape')
                if batch_shape is not None and len(batch_shape) > 0:
                    batch_size = batch_shape[0]
                    input_shape = batch_shape[1:] if len(batch_shape) > 1 else None
            
            # Handle batch_input_shape
            if 'batch_input_shape' in kwargs:
                batch_input_shape = kwargs.pop('batch_input_shape')
                if batch_input_shape is not None and len(batch_input_shape) > 0:
                    batch_size = batch_input_shape[0]
                    input_shape = batch_input_shape[1:] if len(batch_input_shape) > 1 else None
            
            # Remove any other unknown kwargs
            kwargs.pop('type_spec', None)
            
            # Call original with cleaned parameters
            return original_input_layer_init(
                self, input_shape=input_shape, batch_size=batch_size,
                dtype=dtype, input_tensor=input_tensor, sparse=sparse,
                ragged=ragged, name=name
            )
        
        # Apply patch
        tf.keras.layers.InputLayer.__init__ = patched_input_layer_init
        logger.info("✅ TensorFlow compatibility patch applied")
        return True
    except Exception as e:
        logger.error(f"⚠️  Could not apply compatibility patch: {e}")
        return False

def load_model_safe():
    """Load model with compatibility handling"""
    global model
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        return False
    
    try:
        # Apply compatibility patch
        patch_tensorflow_compatibility()
        
        # Try loading with compile=False
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("✅ Model loaded successfully!")
        logger.info(f"   Input shape: {model.input_shape}")
        logger.info(f"   Output shape: {model.output_shape}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        
        # Try alternative method
        try:
            logger.info("Trying alternative loading method...")
            import h5py
            
            # Load with custom object scope
            with tf.keras.utils.custom_object_scope({}):
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
            logger.info("✅ Model loaded with alternative method!")
            return True
        except Exception as e2:
            logger.error(f"❌ Alternative method also failed: {e2}")
            return False

# Load model at startup
model_loaded = load_model_safe()

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

# Detailed snake information database
snake_info_db = {
    "Common Krait": {
        "scientific_name": "Bungarus caeruleus",
        "venomous": True,
        "venom_type": "Neurotoxic",
        "danger_level": "Extremely Dangerous",
        "description": "The Common Krait is a highly venomous nocturnal snake found throughout the Indian subcontinent. It has distinctive white crossbands on a bluish-black body.",
        "venom_effects": "Causes progressive paralysis, respiratory failure, and can be fatal if untreated. Symptoms may not appear immediately but can develop over 4-8 hours.",
        "first_aid": [
            "Keep the victim calm and immobilize the affected limb",
            "Remove any jewelry or tight clothing near the bite",
            "Do NOT apply a tourniquet or try to suck out venom",
            "Get to a hospital IMMEDIATELY - antivenin is critical",
            "Note the time of bite and snake appearance for medical team"
        ],
        "precautions": "Extremely dangerous at night. Never attempt to handle. Most bites occur when people step on them while sleeping on the ground.",
        "habitat": "Found in agricultural areas, scrublands, and human habitations"
    },
    "Banded Krait": {
        "scientific_name": "Bungarus fasciatus",
        "venomous": True,
        "venom_type": "Neurotoxic",
        "danger_level": "Extremely Dangerous",
        "description": "The Banded Krait has distinctive yellow and black bands across its body. It's a nocturnal species that becomes more aggressive at night.",
        "venom_effects": "Powerful neurotoxins cause respiratory paralysis, abdominal pain, and potentially death. Effects can be delayed by several hours.",
        "first_aid": [
            "Immediately immobilize the victim and keep them calm",
            "Apply pressure-immobilization bandage if trained",
            "Rush to nearest hospital with antivenin facility",
            "Monitor breathing - be prepared for CPR",
            "Do not waste time with traditional remedies"
        ],
        "precautions": "Avoid at all costs. Very dangerous, especially at night. Never try to capture or kill the snake.",
        "habitat": "Forests, agricultural lands, and near water bodies"
    },
    "Russell's Viper": {
        "scientific_name": "Daboia russelii",
        "venomous": True,
        "venom_type": "Hemotoxic (affects blood and tissue)",
        "danger_level": "Extremely Dangerous",
        "description": "Russell's Viper is responsible for the most snakebite deaths in India. It has a distinctive chain-like pattern on its back and is known for its aggressive behavior when threatened.",
        "venom_effects": "Causes severe pain, swelling, bleeding disorders, kidney failure, and necrosis. Can be fatal within hours if untreated.",
        "first_aid": [
            "Keep victim still and calm to slow venom spread",
            "Immobilize the bitten limb at heart level",
            "Get to hospital URGENTLY - this is life-threatening",
            "Bring antivenin is critical within first few hours",
            "Monitor for shock and internal bleeding"
        ],
        "precautions": "Stand still if encountered - it strikes when threatened. Responsible for most agricultural snakebites in India. Very aggressive when disturbed.",
        "habitat": "Open grasslands, agricultural fields, and scrublands"
    },
    "Spectacled Cobra": {
        "scientific_name": "Naja naja",
        "venomous": True,
        "venom_type": "Neurotoxic (with cardiotoxic effects)",
        "danger_level": "Extremely Dangerous",
        "description": "The iconic Indian Cobra with a distinctive hood marking resembling spectacles. One of the 'Big Four' venomous snakes in India.",
        "venom_effects": "Causes rapid onset of drowsiness, neurological problems, respiratory failure, and cardiac arrest. Can be fatal within hours.",
        "first_aid": [
            "Immediately get medical help - time is critical",
            "Keep the victim calm and still",
            "Remove constrictive items near the bite",
            "Do NOT apply ice or cut the wound",
            "Antivenin must be administered quickly in hospital"
        ],
        "precautions": "Gives warning by raising hood and hissing. Back away slowly if encountered. Never corner or provoke a cobra.",
        "habitat": "Found near human habitations, agricultural areas, and forests"
    },
    "King Cobra": {
        "scientific_name": "Ophiophagus hannah",
        "venomous": True,
        "venom_type": "Neurotoxic",
        "danger_level": "Extremely Dangerous",
        "description": "The world's longest venomous snake, reaching up to 18 feet. Despite its size, it's generally shy and avoids humans unless threatened or protecting a nest.",
        "venom_effects": "Delivers large quantities of potent neurotoxin. Causes severe pain, blurred vision, drowsiness, and respiratory paralysis. Can kill within 30 minutes.",
        "first_aid": [
            "This is a MEDICAL EMERGENCY - immediate hospitalization required",
            "Keep victim absolutely still and calm",
            "Immobilize the affected area",
            "Prepare for possible respiratory support",
            "Specialized King Cobra antivenin may be needed"
        ],
        "precautions": "Usually avoids humans but extremely dangerous when protecting nest. Can raise one-third of its body off ground. Respect its space and retreat slowly.",
        "habitat": "Dense forests, bamboo thickets, and near streams"
    },
    "Rat Snake": {
        "scientific_name": "Ptyas mucosa",
        "venomous": False,
        "venom_type": "Non-venomous",
        "danger_level": "Harmless",
        "description": "The Oriental Rat Snake is a large, non-venomous snake commonly found near human habitations. It's an excellent climber and is often mistaken for a cobra.",
        "venom_effects": "None - this snake is completely non-venomous and harmless to humans.",
        "first_aid": [
            "Clean any bite wound with soap and water",
            "Apply antiseptic to prevent infection",
            "Monitor for signs of infection over next few days",
            "No antivenin needed - this is a non-venomous species",
            "Tetanus shot may be recommended if wound is deep"
        ],
        "precautions": "Harmless and beneficial - controls rodent populations. May bite defensively but poses no serious threat. Can be safely relocated by professionals if necessary.",
        "habitat": "Human habitations, agricultural areas, trees, and grasslands"
    },
    "Bamboo Pit Viper": {
        "scientific_name": "Trimeresurus gramineus",
        "venomous": True,
        "venom_type": "Hemotoxic",
        "danger_level": "Moderately Dangerous",
        "description": "A beautiful bright green arboreal snake with a triangular head. Generally not aggressive but bites can occur when accidentally touched.",
        "venom_effects": "Causes local pain, swelling, and tissue damage. Rarely fatal but can cause significant local necrosis and complications.",
        "first_aid": [
            "Keep victim calm and immobilized",
            "Clean the wound gently",
            "Seek medical attention for antivenin and monitoring",
            "Watch for excessive swelling or tissue damage",
            "Hospital observation recommended for 24-48 hours"
        ],
        "precautions": "Watch for them in trees and bushes. Bites usually occur when reaching into vegetation. Not aggressive but will bite if touched.",
        "habitat": "Trees, bushes, and vegetation in forests and gardens"
    },
    "Malabar Pit Viper": {
        "scientific_name": "Trimeresurus malabaricus",
        "venomous": True,
        "venom_type": "Hemotoxic",
        "danger_level": "Moderately Dangerous",
        "description": "Found in the Western Ghats, this pit viper has variable coloration ranging from brown to gray with darker markings.",
        "venom_effects": "Causes severe local pain, swelling, bleeding, and tissue necrosis. Rarely fatal but can cause permanent tissue damage if untreated.",
        "first_aid": [
            "Immobilize the affected limb immediately",
            "Keep patient calm and at rest",
            "Get to hospital for antivenin treatment",
            "Monitor for shock and excessive bleeding",
            "Medical supervision required for at least 24 hours"
        ],
        "precautions": "Found in leaf litter and low vegetation. Watch where you step in forested areas. Bites are painful but rarely life-threatening with proper treatment.",
        "habitat": "Western Ghats forests, coffee plantations, and hill stations"
    }
}

def prepare_image(img_path, target_size=(224, 224)):
    """Prepare image for prediction"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "service": "Snake Detector API",
        "version": "1.0"
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please wait for server initialization or contact support."
        }), 503
        
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
        logger.info(f"📸 Processing image: {file.filename}")
        
        img_array = prepare_image(file_path)
        predictions = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx] * 100)
        predicted_species = class_labels[pred_idx]

        logger.info(f"✅ Prediction: {predicted_species} ({confidence:.2f}%)")

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

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host="0.0.0.0", port=port, debug=False)