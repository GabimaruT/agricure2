# 🌿 AgriCure 2.0 — Smart Farming Assistant

> AI-powered crop disease detection platform that combines **image-based diagnosis**, **environmental risk prediction**, and **nearest Krushi Kendra locator** to help Indian farmers protect their crops.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [ML Models](#ml-models)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
- [Training the Models](#training-the-models)
- [Deployment](#deployment)
- [Supported Crops & Diseases](#supported-crops--diseases)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

AgriCure 2.0 is a full-stack agricultural intelligence platform designed for Indian farmers. It provides:

1. **Leaf Scan** — Upload or capture a photo of a diseased leaf and get an instant CNN-based diagnosis with natural & chemical treatment recommendations.
2. **Early Detection** — Enter environmental conditions (temperature, humidity, rainfall, crop type, region, etc.) and receive an ML-driven disease risk assessment before symptoms appear.
3. **Krushi Kendra Finder** — Locate the nearest government agricultural centres (KVKs) across India with distance, phone, and direction links.
4. **Weather Dashboard** — Real-time weather data for any Indian city with agriculture-specific alerts.
5. **Community Forum** — Firebase-backed discussion forum for farmers to share experiences and advice.
6. **Scan History** — Persistent history of past diagnoses stored in Firestore.

---

## Features

| Feature | Description |
|---------|-------------|
| 🔬 **Image Disease Detector** | MobileNetV2 CNN classifies leaf images into 38 disease/healthy classes with **99.35% accuracy** |
| 🌡️ **Environmental Predictor** | XGBoost model predicts disease risk from weather & field conditions with **88.91% accuracy** across 39 classes |
| 🏥 **Krushi Kendra Locator** | Haversine-based nearest-centre search across 21+ KVKs in Maharashtra, Punjab, MP, UP, Karnataka, Telangana, WB & Rajasthan |
| 🌦️ **Weather Integration** | OpenWeatherMap API with agriculture-specific disease risk alerts |
| 💬 **Community Forum** | Real-time Firebase Firestore-backed Q&A with tagging, likes, and replies |
| 🔐 **User Authentication** | Firebase Auth with email/password sign-up, sign-in, and profile management |
| 🌐 **Multilingual** | English & Hindi language toggle in the UI |
| 🎨 **Modern UI** | Dark theme, glassmorphism, micro-animations, Three.js background, fully responsive |
| 🩺 **Natural + Chemical Cures** | Comprehensive treatment database with natural remedies, chemical treatments, and prevention tips |
| 📊 **Risk Assessment** | 4-level risk categorisation: LOW → MEDIUM → HIGH → CRITICAL |

---

## Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.10** | Server runtime |
| **Flask 3.0** | REST API framework |
| **Flask-CORS** | Cross-origin request handling |
| **PyTorch 2.2 (CPU)** | Image model inference (MobileNetV2) |
| **TorchVision 0.17** | Image preprocessing & pretrained models |
| **XGBoost** | Environmental disease prediction |
| **scikit-learn** | Label encoding & metrics |
| **Pandas / NumPy** | Data processing |
| **Pillow** | Image manipulation |
| **Gunicorn** | Production WSGI server |

### Frontend
| Technology | Purpose |
|------------|---------|
| **Vanilla HTML/CSS/JS** | Single-page application (no build step) |
| **Firebase Auth** | User authentication |
| **Firebase Firestore** | Forum posts, scan history persistence |
| **Three.js** | 3D animated background on hero section |
| **Google Fonts** | Syne + DM Sans typography |
| **OpenWeatherMap API** | Live weather data |

### Deployment
| Platform | Role |
|----------|------|
| **Railway** | Backend API server (Nixpacks builder) |
| **Vercel** | Frontend static hosting + serverless proxy |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Vercel)                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  index.html — Single Page Application                 │  │
│  │  ├─ Hero + Three.js background                        │  │
│  │  ├─ Leaf Scan (Image Upload → base64 → API)           │  │
│  │  ├─ Early Detection (Form → API)                      │  │
│  │  ├─ Weather Dashboard (OpenWeatherMap)                 │  │
│  │  ├─ Community Forum (Firebase Firestore)               │  │
│  │  └─ Scan History (Firebase Firestore)                  │  │
│  └───────────────────────────────────────────────────────┘  │
│              │ POST /api/predict-image                       │
│              │ POST /api/predict-env                         │
│              ▼                                               │
│  ┌───────────────────────┐                                  │
│  │  api/analyze.js       │ ← Vercel serverless proxy        │
│  │  (forwards to Railway)│                                  │
│  └───────────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND (Railway)                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  server.py — Flask API                                │  │
│  │  ├─ POST /api/predict-env    → XGBoost model          │  │
│  │  ├─ POST /api/predict-image  → MobileNetV2 CNN        │  │
│  │  ├─ GET  /api/krushi-kendra  → KVK database           │  │
│  │  └─ GET  /health             → Status check            │  │
│  └───────────────────────────────────────────────────────┘  │
│              │                       │                       │
│  ┌───────────┴──────┐   ┌───────────┴──────────────────┐   │
│  │  Model 1          │   │  Model 2                      │   │
│  │  XGBoost (.pkl)   │   │  MobileNetV2 (.pth)           │   │
│  │  88.91% accuracy  │   │  99.35% accuracy              │   │
│  │  39 disease classes│   │  38 disease classes            │   │
│  │  19 features       │   │  224×224 input                 │   │
│  └───────────────────┘   └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Agricure2/
├── server.py                    # Root-level Flask server (standalone mode)
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python version (3.10)
├── Procfile                     # Railway/Heroku start command
├── railway.json                 # Railway deployment config
├── vercel.json                  # Vercel deployment config
├── index.html                   # Root-level frontend (standalone)
│
├── backend/                     # Backend package (deployed to Railway)
│   ├── server.py                # Main Flask API server
│   ├── requirements.txt         # Backend-specific dependencies
│   ├── data/                    # Data files used by backend
│   │   ├── disease_info.json    # Disease metadata (severity, cures)
│   │   ├── natural_cures.json   # Treatment database (natural/chemical/prevention)
│   │   ├── disease_prediction_dataset.csv  # Training dataset (~14 MB)
│   │   └── plantvillage/        # PlantVillage image dataset (gitignored)
│   └── model/                   # Trained model artifacts
│       ├── predict_model.pkl    # XGBoost environmental model (~46 MB)
│       ├── predict_encoders.pkl # Label encoders for categorical features
│       ├── predict_metadata.json# Model accuracy & class info
│       ├── image_model.pth      # MobileNetV2 weights (~9 MB)
│       ├── image_classes.json   # 38 image classification classes
│       └── image_metadata.json  # Image model accuracy & metadata
│
├── frontend/                    # Frontend SPA (deployed to Vercel)
│   └── index.html               # Complete single-page application (128 KB)
│
├── api/                         # Vercel serverless functions
│   └── analyze.js               # Proxy: forwards requests to Railway backend
│
├── ml/                          # ML training & data scripts
│   ├── train_predict.py         # Train XGBoost environmental predictor
│   ├── train_image.py           # Train MobileNetV2 image classifier
│   ├── prepare_real_data.py     # ETL pipeline for raw datasets
│   ├── improve_dataset.py       # Feature engineering (adds 9 new features)
│   └── create_missing_files.py  # Generate disease_info.json & natural_cures.json
│
├── model/                       # Root-level model artifacts (standalone mode)
│   └── (same structure as backend/model/)
│
└── data/                        # Root-level data files (standalone mode)
    ├── disease_info.json
    ├── natural_cures.json
    ├── disease_prediction_dataset.csv
    ├── plantvillage/             # Image dataset (gitignored)
    └── raw/                     # Raw downloaded datasets (gitignored)
```

---

## ML Models

### Model 1: Environmental Disease Predictor (XGBoost)

Predicts crop disease risk based on environmental and agricultural parameters.

| Property | Value |
|----------|-------|
| **Algorithm** | XGBoost (GPU-accelerated when available) |
| **Accuracy** | 88.91% |
| **Classes** | 39 diseases across 15 crops |
| **Features** | 19 engineered features |
| **Training script** | `ml/train_predict.py` |

**Input Features:**
| Feature | Type | Description |
|---------|------|-------------|
| `temperature` | float | Current temperature (°C) |
| `humidity` | float | Relative humidity (%) |
| `rainfall` | float | Rainfall (mm) |
| `wind_speed` | float | Wind speed (km/h) |
| `month` | int | Month of year (1–12) |
| `latitude` / `longitude` | float | GPS coordinates |
| `pesticide_applications` | int | Number of recent pesticide applications |
| `past_disease_occurrences` | int | Historical disease count |
| `crop` | string | Crop name (encoded) |
| `region` | string | District/city (encoded) |
| `state` | string | Indian state (encoded) |
| *Derived* | — | `temp_humidity_index`, `is_monsoon`, `is_winter`, `is_summer`, `high_humidity`, `high_rainfall`, `pesticide_effectiveness` |

### Model 2: Image Disease Detector (MobileNetV2)

Identifies crop diseases from leaf photographs using transfer learning.

| Property | Value |
|----------|-------|
| **Architecture** | MobileNetV2 (pretrained on ImageNet) |
| **Accuracy** | 99.35% |
| **Classes** | 38 (diseases + healthy) |
| **Input size** | 224 × 224 px (RGB) |
| **Dataset** | PlantVillage |
| **Training script** | `ml/train_image.py` |
| **Preprocessing** | Resize → ToTensor → Normalize (ImageNet stats) |
| **Augmentation** | Random horizontal flip, rotation (±15°), colour jitter |

---

## API Reference

Base URL: `http://localhost:5000` (local) or your Railway deployment URL.

### `POST /api/predict-env` — Environmental Disease Prediction

**Request Body (JSON):**
```json
{
  "temperature": 18,
  "humidity": 85,
  "rainfall": 25,
  "wind_speed": 5,
  "month": 7,
  "crop": "Tomato",
  "region": "Amravati",
  "state": "Maharashtra",
  "latitude": 20.93,
  "longitude": 77.75,
  "pesticide_applications": 0,
  "past_disease_occurrences": 1
}
```

**Response:**
```json
{
  "success": true,
  "model": "environmental",
  "risk_level": "HIGH",
  "predictions": [
    {
      "disease": "Tomato___Late_blight",
      "disease_display": "Tomato — Late Blight",
      "probability": 72.3,
      "severity": 65.1,
      "natural_cure": "Bordeaux mixture...",
      "precaution": "Use resistant varieties...",
      "pesticide_effectiveness": 0.7
    }
  ],
  "krushi_kendras": [ ... ],
  "input_summary": { ... }
}
```

### `POST /api/predict-image` — Image-Based Disease Detection

**Request Body (JSON):**
```json
{
  "image": "<base64-encoded-image>",
  "latitude": 20.93,
  "longitude": 77.75
}
```

**Response:**
```json
{
  "success": true,
  "isPlant": true,
  "model": "image_cnn",
  "disease": "Tomato___Late_blight",
  "disease_display": "Tomato — Late Blight",
  "crop": "Tomato",
  "confidence": 94.2,
  "is_healthy": false,
  "severity": 84.8,
  "natural_cure": "Bordeaux mixture...",
  "chemical_treatment": "Apply Mancozeb...",
  "prevention": "Use resistant varieties...",
  "alternatives": [ ... ],
  "krushi_kendras": [ ... ]
}
```

### `GET /api/krushi-kendra` — Find Nearest Krushi Kendras

**Query Parameters:**
| Param | Type | Description |
|-------|------|-------------|
| `city` | string | City name (geocoded via OpenWeatherMap) |
| `lat` | float | Latitude (alternative to city) |
| `lon` | float | Longitude (alternative to city) |

**Example:** `/api/krushi-kendra?city=Nagpur`

**Response:**
```json
{
  "success": true,
  "krushi_kendras": [
    {
      "name": "Krushi Kendra Nagpur",
      "city": "Nagpur",
      "state": "Maharashtra",
      "lat": 21.1458,
      "lon": 79.0882,
      "phone": "0712-2560333",
      "address": "Seminary Hills, Nagpur",
      "distance_km": 3.2
    }
  ],
  "lat": 21.1458,
  "lon": 79.0882
}
```

### `GET /health` — Server Health Check

**Response:**
```json
{
  "status": "ok",
  "models": {
    "environmental": true,
    "image": true
  },
  "env_accuracy": 88.91,
  "image_accuracy": 99.35,
  "image_classes": 38,
  "krushi_kendras": 21
}
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/GabimaruT/agricure2.git
cd agricure2
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is installed as CPU-only (`torch==2.2.0+cpu`) to keep the deployment lightweight. For GPU training, install the CUDA version separately.

### 4. Run the Server

```bash
# Option A: From root (uses server.py + model/ + data/)
python server.py

# Option B: From backend/ directory (uses backend/server.py + backend/model/ + backend/data/)
python backend/server.py
```

The server starts at `http://localhost:5000`.

### 5. Open the Frontend

Navigate to `http://localhost:5000` in your browser — the Flask server serves the frontend directly.

---

## Training the Models

### Environmental Predictor (XGBoost)

```bash
cd ml

# Step 1 (optional): Prepare raw data from Kaggle/ICRISAT
python prepare_real_data.py

# Step 2 (optional): Add extra engineered features
python improve_dataset.py

# Step 3: Train the model
python train_predict.py
```

The trained model is saved to `model/predict_model.pkl`.

### Image Classifier (MobileNetV2)

1. Download the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract it.
2. Update the `data_dir` path in `ml/train_image.py` → `CONFIG['data_dir']`.
3. Run:

```bash
cd ml
python train_image.py
```

The trained model is saved to `model/image_model.pth`.

### Generate Supporting Data Files

```bash
cd ml
python create_missing_files.py
```

This creates `data/disease_info.json` and `data/natural_cures.json`.

---

## Deployment

### Railway (Backend)

1. Push the repo to GitHub.
2. Connect the repository to [Railway](https://railway.app).
3. Railway auto-detects the `Procfile` and `railway.json`:
   - **Build:** Nixpacks
   - **Start command:** `python backend/server.py`
4. Set environment variables (optional):
   - `PORT` — Server port (default: `5000`)
   - `WEATHER_API_KEY` — OpenWeatherMap API key

### Render (Backend — Recommended)

**Option A: One-click Blueprint**

1. Push the repo to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect your repo — Render auto-reads `render.yaml` and configures everything.
4. (Optional) Set `WEATHER_API_KEY` in the environment variables.

**Option B: Manual Setup**

1. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Web Service**.
2. Connect your GitHub repo and configure:

   | Setting | Value |
   |---------|-------|
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `python backend/server.py` |

3. Under **Environment**, add:
   - `PORT` = `5000` (Render also injects this automatically)
   - `PYTHON_VERSION` = `3.10.12`
   - `WEATHER_API_KEY` = your OpenWeatherMap key *(optional)*

4. Choose **Starter** plan or higher (512 MB+ RAM recommended for ML models).

> ⚠️ **Memory Note:** The XGBoost model (~46 MB) + MobileNetV2 (~9 MB) + PyTorch runtime require ~400–500 MB RAM. Render's free tier (512 MB) may be tight — upgrade to Starter if you experience OOM crashes.

Your backend will be live at `https://your-app.onrender.com`.

### Vercel (Frontend)

1. Connect the same repo to [Vercel](https://vercel.com).
2. Vercel serves `frontend/index.html` as static content.
3. The `api/analyze.js` serverless function proxies image prediction requests to the backend.
4. Set environment variable:
   - `ML_SERVER_URL` — Your backend URL (e.g., `https://your-app.onrender.com` or `https://your-app.up.railway.app`)

---

## Supported Crops & Diseases

### 15 Crops

Apple · Blueberry · Cherry · Corn (Maize) · Grape · Orange · Peach · Pepper (Bell) · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato · Wheat

### 38 Disease Classes (Image Model)

| Crop | Diseases |
|------|----------|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn** | Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `5000` | Server port |
| `WEATHER_API_KEY` | No | Built-in key | OpenWeatherMap API key |
| `ML_SERVER_URL` | Vercel only | — | Railway backend URL for the Vercel proxy |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## License

This project is open source. Please check the repository for license details.

---

<p align="center">
  Made with 🌱 for Indian farmers
</p>
