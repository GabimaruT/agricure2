"""
AgriCure 2.0 — Environmental Disease Predictor
GPU-accelerated using XGBoost
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🌱 AgriCure 2.0 — GPU Environmental Disease Predictor")
print("=" * 60)

# ── Install XGBoost if needed ──
try:
    import xgboost as xgb
    print("✅ XGBoost available")
except:
    print("Installing XGBoost...")
    import subprocess
    subprocess.run(["pip", "install", "xgboost"])
    import xgboost as xgb

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    USE_GPU = True
else:
    print("⚠️ No GPU — using CPU")
    USE_GPU = False

# ── LOAD DATA ──
print("\n📂 Loading dataset...")
df = pd.read_csv("../data/disease_prediction_dataset.csv")
print(f"✅ Loaded {len(df)} samples, {df['disease'].nunique()} classes")

# ── ENCODE ──
print("\n🔧 Engineering features...")
le_crop    = LabelEncoder()
le_region  = LabelEncoder()
le_state   = LabelEncoder()
le_disease = LabelEncoder()

df['crop_enc']    = le_crop.fit_transform(df['crop'])
df['region_enc']  = le_region.fit_transform(df['region'])
df['state_enc']   = le_state.fit_transform(df['state'])
df['disease_enc'] = le_disease.fit_transform(df['disease'])

# Derived features
df['temp_humidity_index']    = df['temperature'] * df['humidity'] / 100
df['is_monsoon']             = df['month'].apply(lambda m: 1 if 6<=m<=9 else 0)
df['is_winter']              = df['month'].apply(lambda m: 1 if m<=2 or m>=11 else 0)
df['is_summer']              = df['month'].apply(lambda m: 1 if 3<=m<=5 else 0)
df['high_humidity']          = (df['humidity'] > 75).astype(int)
df['high_rainfall']          = (df['rainfall'] > 20).astype(int)
df['pesticide_effectiveness'] = np.clip(df['pesticide_applications'] * 0.2, 0, 1)

FEATURE_COLS = [
    'temperature','humidity','rainfall','wind_speed',
    'month','latitude','longitude',
    'pesticide_applications','past_disease_occurrences',
    'crop_enc','region_enc','state_enc',
    'temp_humidity_index','is_monsoon','is_winter','is_summer',
    'high_humidity','high_rainfall','pesticide_effectiveness'
]

X = df[FEATURE_COLS]
y = df['disease_enc']

print(f"✅ Features: {len(FEATURE_COLS)}")
print(f"✅ Classes: {len(le_disease.classes_)}")

# ── SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ── TRAIN XGBoost on GPU ──
print(f"\n🚀 Training XGBoost on {'GPU' if USE_GPU else 'CPU'}...")

if USE_GPU:
   model = xgb.XGBClassifier(
    n_estimators=500,      # ← was 300
    max_depth=10,          # ← was 8
    learning_rate=0.05,    # ← was 0.1
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,    # ← add this
    gamma=0.1,             # ← add this
    use_label_encoder=False,
    eval_metric='mlogloss',
    device='cuda',
    random_state=42,
    verbosity=1
)
else:
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        device='cpu',
        random_state=42,
        verbosity=1
    )

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"\n✅ XGBoost Accuracy: {acc*100:.2f}%")

# ── CLASSIFICATION REPORT ──
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le_disease.classes_,
    zero_division=0
))

# ── SAVE ──
print("\n💾 Saving model and encoders...")
os.makedirs("../model", exist_ok=True)

with open("../model/predict_model.pkl", "wb") as f:
    pickle.dump(model, f)

encoders = {
    'le_crop': le_crop, 'le_region': le_region,
    'le_state': le_state, 'le_disease': le_disease,
    'feature_cols': FEATURE_COLS
}
with open("../model/predict_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

metadata = {
    'accuracy': round(acc * 100, 2),
    'model_type': 'XGBoost_GPU' if USE_GPU else 'XGBoost_CPU',
    'n_features': len(FEATURE_COLS),
    'n_classes': len(le_disease.classes_),
    'disease_classes': list(le_disease.classes_),
    'crop_classes': list(le_crop.classes_),
    'region_classes': list(le_region.classes_),
    'feature_cols': FEATURE_COLS,
}
with open("../model/predict_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved!")
print(f"\n🎉 Training Complete! Accuracy: {acc*100:.2f}%")