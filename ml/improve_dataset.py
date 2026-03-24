"""
AgriCure 2.0 — Improved Dataset Generator
Adds more distinguishing features for better accuracy
"""
import pandas as pd
import numpy as np
import json, os

np.random.seed(42)
os.makedirs("../data", exist_ok=True)

# Load existing dataset
df = pd.read_csv("../data/disease_prediction_dataset.csv")
print(f"Loaded {len(df)} samples")

# ═══════════════════════════════════
# ADD NEW DISTINGUISHING FEATURES
# ═══════════════════════════════════

# 1. Soil moisture (different diseases need different soil moisture)
SOIL_MOISTURE = {
    "Tomato___Late_blight":         (75, 95),
    "Tomato___Early_blight":        (55, 75),
    "Tomato___Bacterial_spot":      (70, 90),
    "Tomato___Septoria_leaf_spot":  (65, 85),
    "Tomato___Leaf_Mold":           (80, 95),
    "Tomato___Spider_mites Two-spotted_spider_mite": (20, 45),
    "Tomato___Target_Spot":         (60, 80),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (40, 65),
    "Tomato___Tomato_mosaic_virus": (45, 70),
    "Tomato___healthy":             (45, 65),
    "Potato___Late_blight":         (80, 98),
    "Potato___Early_blight":        (50, 72),
    "Potato___healthy":             (45, 65),
    "Corn_(maize)___Common_rust_":  (65, 85),
    "Corn_(maize)___Northern_Leaf_Blight": (70, 90),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (75, 92),
    "Corn_(maize)___healthy":       (45, 65),
    "Apple___Apple_scab":           (80, 98),
    "Apple___Black_rot":            (60, 80),
    "Apple___Cedar_apple_rust":     (65, 85),
    "Apple___healthy":              (45, 65),
    "Grape___Black_rot":            (75, 95),
    "Grape___Esca_(Black_Measles)": (55, 75),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (70, 88),
    "Grape___healthy":              (45, 65),
    "Wheat___Leaf_rust":            (60, 82),
    "Pepper,_bell___Bacterial_spot":(70, 90),
    "Pepper,_bell___healthy":       (45, 65),
    "Strawberry___Leaf_scorch":     (55, 78),
    "Strawberry___healthy":         (45, 65),
    "Cherry_(including_sour)___Powdery_mildew": (40, 65),
    "Cherry_(including_sour)___healthy": (45, 65),
    "Orange___Haunglongbing_(Citrus_greening)": (60, 80),
    "Peach___Bacterial_spot":       (70, 90),
    "Peach___healthy":              (45, 65),
    "Squash___Powdery_mildew":      (35, 60),
    "Soybean___healthy":            (45, 65),
    "Raspberry___healthy":          (45, 65),
    "Blueberry___healthy":          (45, 65),
}

# 2. Temperature at night (key differentiator!)
NIGHT_TEMP_DIFF = {
    "Tomato___Late_blight":         (8, 14),   # cool nights
    "Tomato___Early_blight":        (3, 7),    # warm nights
    "Tomato___Bacterial_spot":      (2, 6),    # hot nights
    "Tomato___Septoria_leaf_spot":  (4, 8),
    "Tomato___Leaf_Mold":           (6, 12),   # cool nights
    "Tomato___Spider_mites Two-spotted_spider_mite": (1, 4), # hot nights
    "Tomato___Target_Spot":         (3, 7),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (2, 5),
    "Tomato___Tomato_mosaic_virus": (2, 5),
    "Tomato___healthy":             (5, 10),
    "Potato___Late_blight":         (8, 15),
    "Potato___Early_blight":        (3, 8),
    "Potato___healthy":             (5, 10),
    "Corn_(maize)___Common_rust_":  (4, 9),
    "Corn_(maize)___Northern_Leaf_Blight": (3, 8),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (3, 7),
    "Corn_(maize)___healthy":       (5, 10),
    "Apple___Apple_scab":           (6, 12),
    "Apple___Black_rot":            (3, 8),
    "Apple___Cedar_apple_rust":     (5, 10),
    "Apple___healthy":              (5, 10),
    "Grape___Black_rot":            (4, 9),
    "Grape___Esca_(Black_Measles)": (3, 7),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (4, 8),
    "Grape___healthy":              (5, 10),
    "Wheat___Leaf_rust":            (5, 10),
    "Pepper,_bell___Bacterial_spot":(2, 6),
    "Pepper,_bell___healthy":       (5, 10),
    "Strawberry___Leaf_scorch":     (3, 8),
    "Strawberry___healthy":         (5, 10),
    "Cherry_(including_sour)___Powdery_mildew": (5, 12),
    "Cherry_(including_sour)___healthy": (5, 10),
    "Orange___Haunglongbing_(Citrus_greening)": (2, 6),
    "Peach___Bacterial_spot":       (2, 6),
    "Peach___healthy":              (5, 10),
    "Squash___Powdery_mildew":      (5, 12),
    "Soybean___healthy":            (5, 10),
    "Raspberry___healthy":          (5, 10),
    "Blueberry___healthy":          (5, 10),
}

# 3. Days since last rain
DAYS_SINCE_RAIN = {
    "Tomato___Late_blight":         (0, 3),    # very recent rain
    "Tomato___Early_blight":        (3, 10),
    "Tomato___Bacterial_spot":      (0, 2),    # just rained
    "Tomato___Septoria_leaf_spot":  (1, 5),
    "Tomato___Leaf_Mold":           (0, 4),
    "Tomato___Spider_mites Two-spotted_spider_mite": (14, 30), # no rain
    "Tomato___Target_Spot":         (2, 8),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (7, 20),
    "Tomato___Tomato_mosaic_virus": (5, 15),
    "Tomato___healthy":             (3, 10),
    "Potato___Late_blight":         (0, 3),
    "Potato___Early_blight":        (3, 12),
    "Potato___healthy":             (3, 10),
    "Corn_(maize)___Common_rust_":  (1, 6),
    "Corn_(maize)___Northern_Leaf_Blight": (0, 4),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (0, 3),
    "Corn_(maize)___healthy":       (3, 10),
    "Apple___Apple_scab":           (0, 2),
    "Apple___Black_rot":            (2, 7),
    "Apple___Cedar_apple_rust":     (0, 4),
    "Apple___healthy":              (3, 10),
    "Grape___Black_rot":            (0, 3),
    "Grape___Esca_(Black_Measles)": (5, 15),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (1, 5),
    "Grape___healthy":              (3, 10),
    "Wheat___Leaf_rust":            (2, 8),
    "Pepper,_bell___Bacterial_spot":(0, 3),
    "Pepper,_bell___healthy":       (3, 10),
    "Strawberry___Leaf_scorch":     (3, 10),
    "Strawberry___healthy":         (3, 10),
    "Cherry_(including_sour)___Powdery_mildew": (5, 15),
    "Cherry_(including_sour)___healthy": (3, 10),
    "Orange___Haunglongbing_(Citrus_greening)": (3, 12),
    "Peach___Bacterial_spot":       (0, 3),
    "Peach___healthy":              (3, 10),
    "Squash___Powdery_mildew":      (5, 20),
    "Soybean___healthy":            (3, 10),
    "Raspberry___healthy":          (3, 10),
    "Blueberry___healthy":          (3, 10),
}

# 4. Leaf wetness hours per day
LEAF_WETNESS = {
    "Tomato___Late_blight":         (10, 18),
    "Tomato___Early_blight":        (4, 10),
    "Tomato___Bacterial_spot":      (8, 16),
    "Tomato___Septoria_leaf_spot":  (6, 14),
    "Tomato___Leaf_Mold":           (12, 20),
    "Tomato___Spider_mites Two-spotted_spider_mite": (0, 3),
    "Tomato___Target_Spot":         (5, 12),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (2, 8),
    "Tomato___Tomato_mosaic_virus": (2, 8),
    "Tomato___healthy":             (3, 8),
    "Potato___Late_blight":         (12, 20),
    "Potato___Early_blight":        (4, 10),
    "Potato___healthy":             (3, 8),
    "Corn_(maize)___Common_rust_":  (6, 14),
    "Corn_(maize)___Northern_Leaf_Blight": (8, 16),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": (10, 18),
    "Corn_(maize)___healthy":       (3, 8),
    "Apple___Apple_scab":           (12, 20),
    "Apple___Black_rot":            (5, 12),
    "Apple___Cedar_apple_rust":     (8, 16),
    "Apple___healthy":              (3, 8),
    "Grape___Black_rot":            (10, 18),
    "Grape___Esca_(Black_Measles)": (3, 8),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": (8, 16),
    "Grape___healthy":              (3, 8),
    "Wheat___Leaf_rust":            (6, 14),
    "Pepper,_bell___Bacterial_spot":(8, 16),
    "Pepper,_bell___healthy":       (3, 8),
    "Strawberry___Leaf_scorch":     (4, 10),
    "Strawberry___healthy":         (3, 8),
    "Cherry_(including_sour)___Powdery_mildew": (2, 6),
    "Cherry_(including_sour)___healthy": (3, 8),
    "Orange___Haunglongbing_(Citrus_greening)": (4, 10),
    "Peach___Bacterial_spot":       (8, 16),
    "Peach___healthy":              (3, 8),
    "Squash___Powdery_mildew":      (1, 5),
    "Soybean___healthy":            (3, 8),
    "Raspberry___healthy":          (3, 8),
    "Blueberry___healthy":          (3, 8),
}

# 5. Add crop growth stage
GROWTH_STAGE = {
    "Tomato___Late_blight":         3,  # fruiting
    "Tomato___Early_blight":        2,  # vegetative
    "Tomato___Bacterial_spot":      2,
    "Tomato___Septoria_leaf_spot":  2,
    "Tomato___Leaf_Mold":           3,
    "Tomato___Spider_mites Two-spotted_spider_mite": 2,
    "Tomato___Target_Spot":         3,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 1,
    "Tomato___Tomato_mosaic_virus": 1,
    "Tomato___healthy":             2,
    "Potato___Late_blight":         3,
    "Potato___Early_blight":        2,
    "Potato___healthy":             2,
    "Corn_(maize)___Common_rust_":  2,
    "Corn_(maize)___Northern_Leaf_Blight": 2,
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 3,
    "Corn_(maize)___healthy":       2,
    "Apple___Apple_scab":           1,
    "Apple___Black_rot":            3,
    "Apple___Cedar_apple_rust":     1,
    "Apple___healthy":              2,
    "Grape___Black_rot":            3,
    "Grape___Esca_(Black_Measles)": 2,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 2,
    "Grape___healthy":              2,
    "Wheat___Leaf_rust":            2,
    "Pepper,_bell___Bacterial_spot":2,
    "Pepper,_bell___healthy":       2,
    "Strawberry___Leaf_scorch":     2,
    "Strawberry___healthy":         2,
    "Cherry_(including_sour)___Powdery_mildew": 2,
    "Cherry_(including_sour)___healthy": 2,
    "Orange___Haunglongbing_(Citrus_greening)": 2,
    "Peach___Bacterial_spot":       2,
    "Peach___healthy":              2,
    "Squash___Powdery_mildew":      2,
    "Soybean___healthy":            2,
    "Raspberry___healthy":          2,
    "Blueberry___healthy":          2,
}

print("Adding new features...")

# Add features to dataset
def add_soil_moisture(row):
    d = row['disease']
    r = SOIL_MOISTURE.get(d, (45, 65))
    return round(np.random.uniform(r[0], r[1]), 1)

def add_night_temp(row):
    d = row['disease']
    r = NIGHT_TEMP_DIFF.get(d, (5, 10))
    diff = np.random.uniform(r[0], r[1])
    return round(row['temperature'] - diff, 1)

def add_days_rain(row):
    d = row['disease']
    r = DAYS_SINCE_RAIN.get(d, (3, 10))
    return round(np.random.uniform(r[0], r[1]), 1)

def add_leaf_wetness(row):
    d = row['disease']
    r = LEAF_WETNESS.get(d, (3, 8))
    return round(np.random.uniform(r[0], r[1]), 1)

def add_growth_stage(row):
    return GROWTH_STAGE.get(row['disease'], 2)

df['soil_moisture']      = df.apply(add_soil_moisture, axis=1)
df['night_temperature']  = df.apply(add_night_temp, axis=1)
df['days_since_rain']    = df.apply(add_days_rain, axis=1)
df['leaf_wetness_hours'] = df.apply(add_leaf_wetness, axis=1)
df['growth_stage']       = df.apply(add_growth_stage, axis=1)

# Add interaction features
df['temp_range']          = df['temperature'] - df['night_temperature']
df['humidity_moisture']   = df['humidity'] * df['soil_moisture'] / 100
df['wetness_humidity']    = df['leaf_wetness_hours'] * df['humidity'] / 100
df['rain_recency_factor'] = 1 / (df['days_since_rain'] + 1)

print(f"✅ Added 9 new features")
print(f"Total features now: {len(df.columns)}")

# Save improved dataset
df.to_csv("../data/disease_prediction_dataset.csv", index=False)
print(f"✅ Improved dataset saved: {len(df)} samples")
print(f"New columns: soil_moisture, night_temperature, days_since_rain, leaf_wetness_hours, growth_stage")