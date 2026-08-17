# 🌿 AgriCure — ML Model Setup Guide

## Overview
This ML model uses **MobileNetV2** (Transfer Learning) trained on the **PlantVillage dataset** to detect 38 plant diseases across 14 crop types.

---

## Step 1 — Install Python Requirements

```bash
pip install -r requirements.txt
```

---

## Step 2 — Download Dataset

1. Go to: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download and extract to: `./dataset/plantvillage/`

The folder structure should look like:
```
dataset/
  plantvillage/
    Apple___Apple_scab/
    Apple___Black_rot/
    Apple___healthy/
    Corn_(maize)___Common_rust_/
    Tomato___Early_blight/
    Tomato___Late_blight/
    ... (38 folders total)
```

---

## Step 3 — Train the Model

```bash
python train_model.py
```

Training takes:
- **CPU**: ~2-3 hours
- **GPU**: ~20-30 minutes

After training, files are saved to `./model/`:
- `plant_disease_model.pth` — trained weights
- `classes.json` — class names

---

## Step 4 — Start the API Server

```bash
python server.py
```

Server runs at: `http://localhost:5000`

Test it:
```bash
curl http://localhost:5000/health
```

---

## Step 5 — Connect to Frontend (Vercel)

### Option A: Deploy ML server to Railway/Render (Free)

1. Push your ML server to GitHub
2. Deploy to **[railway.app](https://railway.app)** (free tier)
3. Get your server URL (e.g. `https://agricure-ml.railway.app`)
4. Add to Vercel environment variables:
```
ML_SERVER_URL = https://agricure-ml.railway.app
```

### Option B: Run locally with ngrok (for testing)

```bash
# Install ngrok from ngrok.com
ngrok http 5000
```
Copy the ngrok URL and add to Vercel as `ML_SERVER_URL`

### Option C: Use Gemini as fallback
If no ML server, the API automatically uses Gemini if `GEMINI_API_KEY` is set in Vercel.

---

## Supported Crops & Diseases (38 Classes)

| Crop | Diseases Detected |
|------|------------------|
| Apple | Scab, Black Rot, Cedar Rust, Healthy |
| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Rice | Brown Spot, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Yellow Leaf Curl, Mosaic Virus, Healthy |
| Wheat | Leaf Rust, Healthy |
| + more | Peach, Pepper, Strawberry, Squash |

---

## API Reference

### POST /predict
```json
Request:
{
  "image": "<base64 encoded image>"
}

Response:
{
  "isPlant": true,
  "disease": "Early Blight",
  "crop": "Tomato",
  "severity": 65,
  "confidence": 91.2,
  "description": "Dark brown spots with concentric rings...",
  "treatments": ["Apply fungicide...", "Remove infected leaves..."],
  "prevention": "Rotate crops every 3 years...",
  "source": "ml_model"
}
```

### GET /health
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": 38,
  "device": "cpu"
}
```
