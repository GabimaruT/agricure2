import json, os

os.makedirs("../data", exist_ok=True)
os.makedirs("../model", exist_ok=True)

natural_cures = {
    "Tomato___Late_blight": {
        "natural": "Bordeaux mixture (copper sulfate + lime). Remove infected plants. Apply wood ash around plants.",
        "chemical": "Apply Mancozeb or Chlorothalonil fungicide URGENTLY. Stop irrigation.",
        "prevention": "Use resistant varieties. Avoid overhead watering. Monitor during cool wet weather."
    },
    "Tomato___Early_blight": {
        "natural": "Neem oil spray (2ml/L every 7 days). Compost tea spray. Remove lower infected leaves.",
        "chemical": "Apply copper-based or Chlorothalonil fungicide every 7 days.",
        "prevention": "Mulch soil. Rotate crops every 3 years. Stake plants for airflow."
    },
    "Tomato___Bacterial_spot": {
        "natural": "Copper soap spray. Hydrogen peroxide (3%) spray. Remove infected leaves.",
        "chemical": "Apply copper-based bactericide. Switch to drip irrigation.",
        "prevention": "Use disease-free seeds. Avoid working with wet plants."
    },
    "Tomato___Septoria_leaf_spot": {
        "natural": "Compost tea spray. Copper soap spray. Remove infected leaves immediately.",
        "chemical": "Apply Mancozeb 75% WP at 7-10 day intervals.",
        "prevention": "Remove infected debris. Avoid wetting foliage. Stake plants."
    },
    "Tomato___Leaf_Mold": {
        "natural": "Neem oil spray. Improve ventilation. Reduce humidity below 85%.",
        "chemical": "Apply Chlorothalonil fungicide.",
        "prevention": "Maintain humidity below 85%. Improve greenhouse ventilation."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "natural": "Neem oil spray. Strong water spray to dislodge mites. Introduce predatory mites.",
        "chemical": "Apply Spinosad or Abamectin insecticide.",
        "prevention": "Maintain humidity. Avoid dusty conditions."
    },
    "Tomato___Target_Spot": {
        "natural": "Neem oil spray. Copper soap spray. Remove infected plant material.",
        "chemical": "Apply copper-based fungicide every 7 days.",
        "prevention": "Avoid overhead irrigation. 2-3 year crop rotation."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "natural": "Remove infected plants. Apply neem oil to control whitefly vector.",
        "chemical": "Control whitefly with systemic insecticide. Remove infected plants.",
        "prevention": "Use virus-resistant varieties. Use reflective mulches to repel whitefly."
    },
    "Tomato___Tomato_mosaic_virus": {
        "natural": "Remove and burn infected plants. Wash hands before handling plants.",
        "chemical": "No chemical cure. Remove infected plants.",
        "prevention": "Disinfect tools with 10% bleach. Use resistant varieties."
    },
    "Tomato___healthy": {
        "natural": "No treatment needed. Preventive neem oil spray monthly.",
        "chemical": "No treatment needed.",
        "prevention": "Regular monitoring. Proper nutrition and irrigation."
    },
    "Potato___Late_blight": {
        "natural": "Bordeaux mixture. Wood ash. Remove and burn infected plants.",
        "chemical": "Apply Metalaxyl + Mancozeb IMMEDIATELY. Stop irrigation.",
        "prevention": "Plant resistant varieties. Stop irrigation if detected."
    },
    "Potato___Early_blight": {
        "natural": "Neem oil spray. Compost tea. Remove infected lower leaves.",
        "chemical": "Apply Mancozeb or Chlorothalonil fungicide.",
        "prevention": "Use certified seed potatoes. 3-year crop rotation."
    },
    "Potato___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Certified seed potatoes and regular monitoring."
    },
    "Corn_(maize)___Common_rust_": {
        "natural": "Baking soda spray (1 tsp/L). Neem oil spray.",
        "chemical": "Apply Propiconazole at early infection stage.",
        "prevention": "Plant rust-resistant hybrids. Early planting."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "natural": "Neem cake soil application. Trichoderma bio-fungicide.",
        "chemical": "Apply Mancozeb 75% WP at 7-day intervals.",
        "prevention": "Use resistant varieties. Crop rotation."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "natural": "Trichoderma spray. Remove infected crop debris.",
        "chemical": "Apply Triazole fungicide (Propiconazole).",
        "prevention": "Improve air circulation. Use resistant varieties."
    },
    "Corn_(maize)___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Maintain proper spacing and nutrition."
    },
    "Apple___Apple_scab": {
        "natural": "Lime sulfur spray during dormant season. Neem oil spray. Remove fallen leaves.",
        "chemical": "Apply Captan or Mancozeb fungicide at 7-10 day intervals.",
        "prevention": "Plant scab-resistant varieties. Apply dormant lime sulfur before bud break."
    },
    "Apple___Black_rot": {
        "natural": "Bordeaux mixture spray. Remove mummified fruits. Apply compost tea.",
        "chemical": "Apply Captan-based fungicide. Prune 15cm below infection.",
        "prevention": "Maintain orchard sanitation. Remove dead wood regularly."
    },
    "Apple___Cedar_apple_rust": {
        "natural": "Neem oil spray. Remove nearby cedar/juniper trees. Apply sulfur dust.",
        "chemical": "Apply Myclobutanil or Triadimefon fungicide.",
        "prevention": "Plant rust-resistant apple varieties. Create distance from cedar trees."
    },
    "Apple___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Maintain proper nutrition, irrigation, and regular monitoring."
    },
    "Grape___Black_rot": {
        "natural": "Bordeaux mixture spray. Remove mummified fruits. Baking soda spray.",
        "chemical": "Apply Myclobutanil or Mancozeb.",
        "prevention": "Vineyard sanitation. Prune for air circulation."
    },
    "Grape___Esca_(Black_Measles)": {
        "natural": "No effective cure. Remove and burn infected vines. Apply wound protectant.",
        "chemical": "Consult extension officer.",
        "prevention": "Protect pruning wounds. Avoid large cuts."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "natural": "Copper sulfate spray. Neem oil. Bordeaux mixture.",
        "chemical": "Apply copper-based fungicide.",
        "prevention": "Improve air circulation in vineyard."
    },
    "Grape___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Regular canopy management and monitoring."
    },
    "Wheat___Leaf_rust": {
        "natural": "Neem oil spray. Wood ash application.",
        "chemical": "Apply Propiconazole or Tebuconazole at first sign.",
        "prevention": "Grow rust-resistant varieties. Early planting."
    },
    "Pepper,_bell___Bacterial_spot": {
        "natural": "Copper soap spray. Garlic extract spray. Remove infected leaves.",
        "chemical": "Apply copper-based bactericide.",
        "prevention": "Use disease-free seeds. Work in dry conditions only."
    },
    "Pepper,_bell___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Maintain proper nutrition and air circulation."
    },
    "Strawberry___Leaf_scorch": {
        "natural": "Neem oil spray. Compost tea. Remove infected leaves.",
        "chemical": "Apply Captan fungicide.",
        "prevention": "Use disease-free plants. Avoid overhead irrigation."
    },
    "Strawberry___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Good drainage and air circulation."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "natural": "Baking soda spray. Milk spray (40% milk). Neem oil.",
        "chemical": "Apply wettable sulfur or Myclobutanil.",
        "prevention": "Prune for open canopy. Avoid excess nitrogen fertilizer."
    },
    "Cherry_(including_sour)___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Regular pruning and monitoring."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "natural": "No cure. Remove and burn infected trees. Apply neem oil to control psyllid.",
        "chemical": "Control psyllid with systemic insecticide.",
        "prevention": "Certified disease-free nursery stock. Control Asian citrus psyllid."
    },
    "Peach___Bacterial_spot": {
        "natural": "Copper soap spray. Garlic extract spray. Remove infected leaves.",
        "chemical": "Apply copper-based bactericide.",
        "prevention": "Plant resistant varieties. Apply copper sprays in autumn."
    },
    "Peach___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Regular monitoring and proper nutrition."
    },
    "Squash___Powdery_mildew": {
        "natural": "Baking soda + neem oil spray. Milk spray (40% milk/60% water).",
        "chemical": "Apply wettable sulfur or Myclobutanil.",
        "prevention": "Avoid dense planting. Water at base only."
    },
    "Soybean___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Balanced fertilization and regular monitoring."
    },
    "Raspberry___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Proper drainage and air circulation."
    },
    "Blueberry___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Maintain acidic soil pH 4.5-5.5. Regular monitoring."
    }
}

with open("../data/natural_cures.json", "w") as f:
    json.dump(natural_cures, f, indent=2)
print("✅ natural_cures.json saved!")

disease_info = {}
for disease, cures in natural_cures.items():
    disease_info[disease] = {
        "crop": disease.split("___")[0].replace("_"," "),
        "natural_cure": cures["natural"],
        "precaution": cures["prevention"],
        "severity_base": 0 if "healthy" in disease else 65
    }

with open("../data/disease_info.json", "w") as f:
    json.dump(disease_info, f, indent=2)
print("✅ disease_info.json saved!")
print("\nAll missing files created! ✅")