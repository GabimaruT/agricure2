"""
AgriCure 2.0 — Main Flask Backend Server
Endpoints:
  POST /api/predict-env    — ML Model 1: Environmental disease prediction
  POST /api/predict-image  — ML Model 2: Image-based disease detection
  GET  /api/krushi-kendra  — Find nearest Krushi Kendra
  GET  /health             — Server health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import base64
import io
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import requests

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════
# LOAD CONFIGURATION
# ═══════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR  = os.path.join(BASE_DIR, 'data')
device = torch.device('cpu')

# ═══════════════════════════════════════
# LOAD ML MODEL 1: ENVIRONMENTAL PREDICTOR
# ═══════════════════════════════════════
predict_model = None
predict_encoders = None
predict_metadata = None

def load_predict_model():
    global predict_model, predict_encoders, predict_metadata
    try:
        model_path    = os.path.join(MODEL_DIR, 'predict_model.pkl')
        encoders_path = os.path.join(MODEL_DIR, 'predict_encoders.pkl')
        metadata_path = os.path.join(MODEL_DIR, 'predict_metadata.json')

        with open(model_path, 'rb') as f:
            predict_model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            predict_encoders = pickle.load(f)
        with open(metadata_path, 'r') as f:
            predict_metadata = json.load(f)

        print(f"✅ Environmental Model loaded! Accuracy: {predict_metadata['accuracy']}%")
        return True
    except Exception as e:
        print(f"⚠️ Environmental model not found: {e}")
        return False

# ═══════════════════════════════════════
# LOAD ML MODEL 2: IMAGE DETECTOR
# ═══════════════════════════════════════
image_model    = None
image_classes  = None
image_metadata = None
natural_cures  = None

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_image_model():
    global image_model, image_classes, image_metadata, natural_cures
    try:
        model_path    = os.path.join(MODEL_DIR, 'image_model.pth')
        classes_path  = os.path.join(MODEL_DIR, 'image_classes.json')
        metadata_path = os.path.join(MODEL_DIR, 'image_metadata.json')
        cures_path    = os.path.join(DATA_DIR, 'natural_cures.json')

        with open(classes_path, 'r') as f:
            image_classes = json.load(f)
        with open(metadata_path, 'r') as f:
            image_metadata = json.load(f)
        with open(cures_path, 'r') as f:
            natural_cures = json.load(f)

        # Build model architecture
        n_classes = len(image_classes)
        image_model = models.mobilenet_v2(pretrained=False)
        image_model.classifier[1] = nn.Linear(image_model.last_channel, n_classes)
        image_model.load_state_dict(
            torch.load(model_path, map_location='cpu')
        )
        image_model.eval()
        print(f"✅ Image Model loaded! {n_classes} classes, Accuracy: {image_metadata.get('accuracy','--')}%")
        return True
    except Exception as e:
        print(f"⚠️ Image model not found: {e}")
        return False

# ═══════════════════════════════════════
# LOAD DISEASE INFO
# ═══════════════════════════════════════
disease_info = {}
def load_disease_info():
    global disease_info
    try:
        with open(os.path.join(DATA_DIR, 'disease_info.json'), 'r') as f:
            disease_info = json.load(f)
        print(f"✅ Disease info loaded: {len(disease_info)} entries")
    except Exception as e:
        print(f"⚠️ Disease info not found: {e}")

# ═══════════════════════════════════════
# KRUSHI KENDRA DATABASE (Maharashtra + Major States)
# ═══════════════════════════════════════
KRUSHI_KENDRA_DB = [
    # Maharashtra
    {"name": "Krushi Vigyan Kendra Amravati", "city": "Amravati", "state": "Maharashtra",
     "lat": 20.9302, "lon": 77.7523, "phone": "0721-2662179", "address": "PDKV Campus, Akola Road, Amravati"},
    {"name": "Zilla Krushi Kendra Amravati", "city": "Amravati", "state": "Maharashtra",
     "lat": 20.9374, "lon": 77.7796, "phone": "0721-2660541", "address": "Near Collector Office, Amravati"},
    {"name": "Krushi Vigyan Kendra Akola", "city": "Akola", "state": "Maharashtra",
     "lat": 20.7002, "lon": 77.0082, "phone": "0724-2435152", "address": "PDKV Campus, Akola"},
    {"name": "Krushi Kendra Nagpur", "city": "Nagpur", "state": "Maharashtra",
     "lat": 21.1458, "lon": 79.0882, "phone": "0712-2560333", "address": "Seminary Hills, Nagpur"},
    {"name": "KVK Nashik", "city": "Nashik", "state": "Maharashtra",
     "lat": 20.0059, "lon": 73.7750, "phone": "0253-2571000", "address": "Dindori Road, Nashik"},
    {"name": "KVK Pune", "city": "Pune", "state": "Maharashtra",
     "lat": 18.5204, "lon": 73.8567, "phone": "020-26056000", "address": "Shivajinagar, Pune"},
    {"name": "KVK Aurangabad", "city": "Aurangabad", "state": "Maharashtra",
     "lat": 19.8762, "lon": 75.3433, "phone": "0240-2332090", "address": "Krishi Nagar, Aurangabad"},
    {"name": "KVK Latur", "city": "Latur", "state": "Maharashtra",
     "lat": 18.4088, "lon": 76.5604, "phone": "02382-244242", "address": "Agricultural College Campus, Latur"},
    {"name": "KVK Nanded", "city": "Nanded", "state": "Maharashtra",
     "lat": 19.1383, "lon": 77.3210, "phone": "02462-234567", "address": "Vasant Nagar, Nanded"},
    {"name": "KVK Kolhapur", "city": "Kolhapur", "state": "Maharashtra",
     "lat": 16.7050, "lon": 74.2433, "phone": "0231-2658000", "address": "Rajaram Colony, Kolhapur"},
    {"name": "KVK Solapur", "city": "Solapur", "state": "Maharashtra",
     "lat": 17.6599, "lon": 75.9064, "phone": "0217-2307000", "address": "Hotgi Road, Solapur"},
    # Punjab
    {"name": "PAU Ludhiana - KVK", "city": "Ludhiana", "state": "Punjab",
     "lat": 30.9010, "lon": 75.8573, "phone": "0161-2401960", "address": "PAU Campus, Ludhiana"},
    {"name": "KVK Amritsar", "city": "Amritsar", "state": "Punjab",
     "lat": 31.6340, "lon": 74.8723, "phone": "0183-2258000", "address": "Tarn Taran Road, Amritsar"},
    # Madhya Pradesh
    {"name": "KVK Indore", "city": "Indore", "state": "Madhya Pradesh",
     "lat": 22.7196, "lon": 75.8577, "phone": "0731-2465000", "address": "JNKVV Campus, Indore"},
    {"name": "KVK Bhopal", "city": "Bhopal", "state": "Madhya Pradesh",
     "lat": 23.2599, "lon": 77.4126, "phone": "0755-2574000", "address": "Berasia Road, Bhopal"},
    # Uttar Pradesh
    {"name": "KVK Lucknow", "city": "Lucknow", "state": "Uttar Pradesh",
     "lat": 26.8467, "lon": 80.9462, "phone": "0522-2740000", "address": "Amausi, Lucknow"},
    {"name": "KVK Varanasi", "city": "Varanasi", "state": "Uttar Pradesh",
     "lat": 25.3176, "lon": 82.9739, "phone": "0542-2362000", "address": "BHU Campus, Varanasi"},
    # Karnataka
    {"name": "UAS Bangalore - KVK", "city": "Bangalore", "state": "Karnataka",
     "lat": 12.9716, "lon": 77.5946, "phone": "080-23331000", "address": "GKVK Campus, Bangalore"},
    # Telangana
    {"name": "KVK Hyderabad", "city": "Hyderabad", "state": "Telangana",
     "lat": 17.3850, "lon": 78.4867, "phone": "040-27152000", "address": "Rajendranagar, Hyderabad"},
    # West Bengal
    {"name": "KVK Kolkata", "city": "Kolkata", "state": "West Bengal",
     "lat": 22.5726, "lon": 88.3639, "phone": "033-25825000", "address": "Bidhan Nagar, Kolkata"},
    # Rajasthan
    {"name": "KVK Jaipur", "city": "Jaipur", "state": "Rajasthan",
     "lat": 26.9124, "lon": 75.7873, "phone": "0141-2710000", "address": "Durgapura, Jaipur"},
]

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in km"""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def get_nearest_krushi_kendras(lat, lon, n=3):
    """Get n nearest Krushi Kendras"""
    for kk in KRUSHI_KENDRA_DB:
        kk['distance_km'] = round(haversine_distance(lat, lon, kk['lat'], kk['lon']), 1)
    return sorted(KRUSHI_KENDRA_DB, key=lambda x: x['distance_km'])[:n]

# ═══════════════════════════════════════
# HELPER: GET COORDINATES FROM CITY
# ═══════════════════════════════════════
def get_city_coordinates(city_name):
    """Get lat/lon from city name using OpenWeatherMap"""
    try:
        WK = os.environ.get('WEATHER_API_KEY', 'bd5e378503939ddaee76f12ad7a97608')
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},IN&limit=1&appid={WK}"
        r = requests.get(url, timeout=5)
        data = r.json()
        if data:
            return data[0]['lat'], data[0]['lon']
    except:
        pass
    # Fallback to Amravati
    return 20.93, 77.75

# ═══════════════════════════════════════
# API ENDPOINT 1: ENVIRONMENTAL PREDICTION
# ═══════════════════════════════════════
@app.route('/api/predict-env', methods=['POST'])
def predict_environmental():
    try:
        data = request.get_json()

        # Extract inputs
        temperature          = float(data.get('temperature', 25))
        humidity             = float(data.get('humidity', 70))
        rainfall             = float(data.get('rainfall', 10))
        wind_speed           = float(data.get('wind_speed', 5))
        month                = int(data.get('month', 6))
        crop                 = data.get('crop', 'Tomato')
        region               = data.get('region', 'Amravati')
        state                = data.get('state', 'Maharashtra')
        latitude             = float(data.get('latitude', 20.93))
        longitude            = float(data.get('longitude', 77.75))
        pesticide_count      = int(data.get('pesticide_applications', 0))
        past_disease_count   = int(data.get('past_disease_occurrences', 0))

        # Get nearest Krushi Kendras
        nearest_kks = get_nearest_krushi_kendras(latitude, longitude, n=3)

        if predict_model is None or predict_encoders is None:
            return _fallback_env_prediction(data, nearest_kks)

        # Prepare features
        le_crop    = predict_encoders['le_crop']
        le_region  = predict_encoders['le_region']
        le_state   = predict_encoders['le_state']
        le_disease = predict_encoders['le_disease']
        feat_cols  = predict_encoders['feature_cols']

        # Encode categorical (handle unseen labels)
        try:
            crop_enc = le_crop.transform([crop])[0]
        except:
            crop_enc = 0
        try:
            region_enc = le_region.transform([region])[0]
        except:
            region_enc = 0
        try:
            state_enc = le_state.transform([state])[0]
        except:
            state_enc = 0

        # Derived features
        temp_hum_idx = temperature * humidity / 100
        is_monsoon   = 1 if 6 <= month <= 9 else 0
        is_winter    = 1 if month <= 2 or month >= 11 else 0
        is_summer    = 1 if 3 <= month <= 5 else 0
        high_hum     = 1 if humidity > 75 else 0
        high_rain    = 1 if rainfall > 20 else 0
        pest_eff     = min(1.0, pesticide_count * 0.2)

        features = [[
            temperature, humidity, rainfall, wind_speed,
            month, latitude, longitude,
            pesticide_count, past_disease_count,
            crop_enc, region_enc, state_enc,
            temp_hum_idx, is_monsoon, is_winter, is_summer,
            high_hum, high_rain, pest_eff
        ]]

        # Predict with probabilities
        proba = predict_model.predict_proba(features)[0]
        top_indices = np.argsort(proba)[::-1][:5]  # top 5

        predictions = []
        for idx in top_indices:
            disease_name = le_disease.classes_[idx]
            prob = float(proba[idx])
            if prob < 0.05 or disease_name == "No_disease":
                continue

            d_info = disease_info.get(disease_name, {})
            severity = d_info.get('severity_base', 60)
            # Adjust severity based on probability
            severity = round(severity * prob * 1.5, 1)
            severity = min(100, severity)

            predictions.append({
                'disease': disease_name,
                'disease_display': disease_name.replace('___', ' — ').replace('_', ' '),
                'probability': round(prob * 100, 1),
                'severity': severity,
                'natural_cure': d_info.get('natural_cure', 'Consult local agricultural officer.'),
                'precaution': d_info.get('precaution', 'Regular monitoring recommended.'),
                'pesticide_effectiveness': d_info.get('pesticide_effect', 0.6),
            })

        # Risk level
        top_prob = float(proba[top_indices[0]]) if len(top_indices) > 0 else 0
        top_disease = le_disease.classes_[top_indices[0]] if len(top_indices) > 0 else "No_disease"

        if top_disease == "No_disease" or top_prob < 0.3:
            risk_level = "LOW"
        elif top_prob < 0.5:
            risk_level = "MEDIUM"
        elif top_prob < 0.75:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return jsonify({
            'success': True,
            'model': 'environmental',
            'risk_level': risk_level,
            'predictions': predictions[:4],  # top 4
            'krushi_kendras': nearest_kks,
            'input_summary': {
                'crop': crop, 'region': region, 'month': month,
                'temp': temperature, 'humidity': humidity,
                'pesticide_applications': pesticide_count,
                'past_disease_count': past_disease_count
            }
        })

    except Exception as e:
        print(f"Environmental prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _fallback_env_prediction(data, nearest_kks):
    """Rule-based fallback when ML model not available"""
    temp     = float(data.get('temperature', 25))
    humidity = float(data.get('humidity', 70))
    rainfall = float(data.get('rainfall', 10))
    month    = int(data.get('month', 6))
    crop     = data.get('crop', 'Tomato')
    pest_count = int(data.get('pesticide_applications', 0))

    predictions = []
    # Simple rule-based prediction
    if crop == 'Tomato' and humidity > 80 and 10 <= temp <= 22:
        predictions.append({
            'disease': 'Tomato___Late_blight',
            'disease_display': 'Tomato — Late Blight',
            'probability': min(95, humidity - 30),
            'severity': 90,
            'natural_cure': 'Bordeaux mixture. Remove infected plants. Apply wood ash.',
            'precaution': 'Use drip irrigation. Apply Mancozeb preventively.',
            'pesticide_effectiveness': 0.7
        })
    elif crop == 'Potato' and humidity > 85 and 10 <= temp <= 20:
        predictions.append({
            'disease': 'Potato___Late_blight',
            'disease_display': 'Potato — Late Blight',
            'probability': min(95, humidity - 25),
            'severity': 95,
            'natural_cure': 'Bordeaux mixture. Remove and burn infected plants.',
            'precaution': 'Plant resistant varieties. Stop irrigation if detected.',
            'pesticide_effectiveness': 0.75
        })

    risk_level = "CRITICAL" if predictions and predictions[0]['probability'] > 75 else \
                 "HIGH" if predictions else "LOW"

    return jsonify({
        'success': True,
        'model': 'rule_based_fallback',
        'risk_level': risk_level,
        'predictions': predictions,
        'krushi_kendras': nearest_kks,
    })


# ═══════════════════════════════════════
# API ENDPOINT 2: IMAGE PREDICTION
# ═══════════════════════════════════════
@app.route('/api/predict-image', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        image_b64 = data.get('image', '')
        latitude  = float(data.get('latitude', 20.93))
        longitude = float(data.get('longitude', 77.75))

        if not image_b64:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        # Decode image
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Get nearest Krushi Kendras
        nearest_kks = get_nearest_krushi_kendras(latitude, longitude, n=3)

        if image_model is None:
            return jsonify({
                'success': False,
                'error': 'Image model not loaded. Please train first.',
                'krushi_kendras': nearest_kks
            }), 503

        # Preprocess
        img_tensor = IMAGE_TRANSFORM(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = image_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_probs, top_indices = torch.topk(probabilities, 3)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            disease_key = image_classes[idx.item()]
            confidence  = float(prob.item()) * 100
            is_healthy  = 'healthy' in disease_key.lower()

            cure_info = natural_cures.get(disease_key, {
                'natural': 'Consult local agricultural officer.',
                'chemical': 'Consult local agricultural officer.',
                'prevention': 'Regular monitoring recommended.'
            })

            crop_name = disease_key.split('___')[0].replace('_', ' ')
            disease_display = disease_key.replace('___', ' — ').replace('_', ' ')

            results.append({
                'disease': disease_key,
                'disease_display': disease_display,
                'crop': crop_name,
                'confidence': round(confidence, 1),
                'is_healthy': is_healthy,
                'severity': 0 if is_healthy else min(95, round(confidence * 0.9, 1)),
                'natural_cure': cure_info.get('natural', ''),
                'chemical_treatment': cure_info.get('chemical', ''),
                'prevention': cure_info.get('prevention', '')
            })

        top = results[0]

        return jsonify({
            'success': True,
            'isPlant': True,
            'model': 'image_cnn',
            'disease': top['disease'],
            'disease_display': top['disease_display'],
            'crop': top['crop'],
            'confidence': top['confidence'],
            'is_healthy': top['is_healthy'],
            'severity': top['severity'],
            'natural_cure': top['natural_cure'],
            'chemical_treatment': top['chemical_treatment'],
            'prevention': top['prevention'],
            'alternatives': results[1:],
            'krushi_kendras': nearest_kks
        })

    except Exception as e:
        print(f"Image prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════
# API ENDPOINT 3: NEAREST KRUSHI KENDRA
# ═══════════════════════════════════════
@app.route('/api/krushi-kendra', methods=['GET'])
def krushi_kendra():
    try:
        city = request.args.get('city', '')
        lat  = request.args.get('lat', None)
        lon  = request.args.get('lon', None)

        if lat and lon:
            latitude  = float(lat)
            longitude = float(lon)
        elif city:
            latitude, longitude = get_city_coordinates(city)
        else:
            return jsonify({'success': False, 'error': 'Provide city or coordinates'}), 400

        nearest = get_nearest_krushi_kendras(latitude, longitude, n=5)
        return jsonify({'success': True, 'krushi_kendras': nearest, 'lat': latitude, 'lon': longitude})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models': {
            'environmental': predict_model is not None,
            'image': image_model is not None,
        },
        'env_accuracy': predict_metadata.get('accuracy') if predict_metadata else None,
        'image_accuracy': image_metadata.get('accuracy') if image_metadata else None,
        'image_classes': len(image_classes) if image_classes else 0,
        'krushi_kendras': len(KRUSHI_KENDRA_DB)
    })


# ═══════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════
print("\n🌿 AgriCure 2.0 Backend Starting...")
print("=" * 50)
load_disease_info()
load_predict_model()
load_image_model()
print("=" * 50)
print("🚀 Server ready!")
print("📡 Endpoints:")
print("   POST /api/predict-env    — Environmental disease prediction")
print("   POST /api/predict-image  — Image-based disease detection")
print("   GET  /api/krushi-kendra  — Find nearest Krushi Kendra")
print("   GET  /health             — Server status")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
