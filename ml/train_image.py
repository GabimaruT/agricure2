"""
AgriCure 2.0 — ML Model 2: Image-Based Disease Detector
Uses: MobileNetV2 Transfer Learning on PlantVillage dataset
Predicts: Disease from leaf image + natural cure suggestions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
import os
import time

print("🔬 AgriCure 2.0 — Image Disease Detector Training")
print("=" * 60)

# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════
CONFIG = {
    'data_dir': 'C:/hackton2/dataset/plantvillage/color',

    'model_save_path': '../model/image_model.pth',
    'classes_save_path': '../model/image_classes.json',
    'image_size': 128,       # ← was 224 (3x faster!)
    'batch_size': 64,        # ← was 32 (2x faster!)
    'num_epochs': 10,        # ← was 15 (saves 5 epochs)
    'learning_rate': 0.001,
    'val_split': 0.2,
    'num_workers': 0,        # ← was 0 (parallel loading!)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"📱 Device: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# ═══════════════════════════════════════
# NATURAL CURE DATABASE
# ═══════════════════════════════════════
NATURAL_CURES = {
    "Apple___Apple_scab": {
        "natural": "Spray lime sulfur during dormant season. Apply neem oil every 7 days. Remove and burn fallen leaves.",
        "chemical": "Apply Captan or Mancozeb fungicide at 7-10 day intervals.",
        "prevention": "Plant scab-resistant varieties. Prune for air circulation."
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
        "natural": "No treatment needed. Continue regular neem oil preventive spray.",
        "chemical": "No treatment needed.",
        "prevention": "Maintain proper nutrition, irrigation, and regular monitoring."
    },
    "Blueberry___healthy": {
        "natural": "No treatment needed.",
        "chemical": "No treatment needed.",
        "prevention": "Maintain acidic soil pH 4.5-5.5. Regular monitoring."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "natural": "Baking soda spray (1 tsp + few drops soap per litre). Milk spray (40% milk). Neem oil.",
        "chemical": "Apply wettable sulfur or Myclobutanil.",
        "prevention": "Prune for open canopy. Avoid excess nitrogen fertilizer."
    },
    "Cherry_(including_sour)___healthy": {
        "natural": "No treatment needed.", "chemical": "No treatment needed.",
        "prevention": "Regular pruning and monitoring."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "natural": "Trichoderma bio-fungicide. Neem cake soil application. Remove infected debris.",
        "chemical": "Apply Triazole fungicide (Propiconazole).",
        "prevention": "Improve air circulation. Use resistant varieties. Crop rotation."
    },
    "Corn_(maize)___Common_rust_": {
        "natural": "Baking soda spray. Neem oil spray. Wood ash application.",
        "chemical": "Apply Propiconazole at early infection stage.",
        "prevention": "Plant rust-resistant hybrids. Early planting."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "natural": "Neem cake soil application. Remove infected plant debris. Trichoderma spray.",
        "chemical": "Apply Mancozeb 75% WP at 7-day intervals.",
        "prevention": "Use resistant varieties. Balanced fertilization."
    },
    "Corn_(maize)___healthy": {
        "natural": "No treatment needed.", "chemical": "No treatment needed.",
        "prevention": "Maintain proper spacing and nutrition."
    },
    "Grape___Black_rot": {
        "natural": "Bordeaux mixture. Remove mummified fruits. Baking soda spray.",
        "chemical": "Apply Myclobutanil or Mancozeb.",
        "prevention": "Vineyard sanitation. Prune for air circulation."
    },
    "Grape___Esca_(Black_Measles)": {
        "natural": "No effective natural cure. Remove and burn infected vines. Apply wound protectant.",
        "chemical": "Consult extension officer. Sodium arsenite (restricted).",
        "prevention": "Protect pruning wounds. Avoid large cuts."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "natural": "Copper sulfate spray. Neem oil. Bordeaux mixture.",
        "chemical": "Apply copper-based fungicide.",
        "prevention": "Improve canopy air circulation."
    },
    "Grape___healthy": {
        "natural": "No treatment needed.", "chemical": "No treatment needed.",
        "prevention": "Regular canopy management and monitoring."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "natural": "No cure. Remove and burn infected trees. Apply neem oil to control psyllid.",
        "chemical": "Control psyllid with systemic insecticide. No chemical cure for HLB.",
        "prevention": "Certified disease-free nursery stock. Control Asian citrus psyllid."
    },
    "Peach___Bacterial_spot": {
        "natural": "Copper soap spray. Garlic extract spray. Remove infected leaves.",
        "chemical": "Apply copper-based bactericide.",
        "prevention": "Plant resistant varieties. Apply copper sprays in autumn."
    },
    "Peach___healthy": {
        "natural": "No treatment needed.", "chemical": "No treatment needed.",
        "prevention": "Regular monitoring and proper nutrition."
    },
    "Pepper,_bell___Bacterial_spot": {
        "natural": "Copper soap spray. Hydrogen peroxide diluted spray. Remove infected parts.",
        "chemical": "Apply copper-based bactericide. Avoid overhead irrigation.",
        "prevention": "Use disease-free seeds. Work in dry conditions only."
    },
    "Pepper,_bell___healthy": {
        "natural": "No treatment needed.", "chemical": "No treatment needed.",
        "prevention": "Maintain proper nutrition and air circulation."
    },
    "Potato___Early_blight": {
        "natural": "Neem oil spray. Compost tea. Remove infected lower leaves.",
        "chemical": "Apply Mancozeb or Chlorothalonil.",
        "prevention": "Use certified seed potatoes. 3-year crop rotation."
    },
    "Potato___Late_blight": {
        "natural": "Bordeaux mixture (copper sulfate + lime). Wood ash. Remove and burn infected plants.",
        "chemical": "Apply Metalaxyl + Mancozeb IMMEDIATELY. Stop irrigation.",
        "prevention": "Plant resistant varieties. Stop irrigation if detected. Destroy volunteers."
    },
    "Potato___healthy": {
        "natural": "No treatment needed.", "chemical": "No treatment needed.",
        "prevention": "Certified seed potatoes and regular monitoring."
    },
    "Raspberry___healthy": {
        "natural": "No treatment needed.", "chemical": "No treatment needed.",
        "prevention": "Proper drainage and air circulation."
    },
    "Soybean___healthy": {
        "natural": "No treatment needed.", "chemical": "No treatment needed.",
        "prevention": "Balanced fertilization and regular monitoring."
    },
    "Squash___Powdery_mildew": {
        "natural": "Baking soda + neem oil spray. Milk spray (40% milk/60% water). Potassium bicarbonate.",
        "chemical": "Apply wettable sulfur or Myclobutanil.",
        "prevention": "Avoid dense planting. Water at base only."
    },
    "Strawberry___Leaf_scorch": {
        "natural": "Neem oil spray. Compost tea. Remove infected leaves immediately.",
        "chemical": "Apply Captan fungicide.",
        "prevention": "Use disease-free plants. Avoid overhead irrigation."
    },
    "Strawberry___healthy": {
        "natural": "No treatment needed.", "chemical": "No treatment needed.",
        "prevention": "Good drainage and air circulation."
    },
    "Tomato___Bacterial_spot": {
        "natural": "Copper soap spray. Garlic extract. Hydrogen peroxide spray (3%).",
        "chemical": "Apply copper-based bactericide. Switch to drip irrigation.",
        "prevention": "Disease-free seeds. Avoid working with wet plants."
    },
    "Tomato___Early_blight": {
        "natural": "Neem oil spray every 7 days. Compost tea. Remove lower infected leaves.",
        "chemical": "Apply copper-based or Chlorothalonil fungicide every 7 days.",
        "prevention": "Mulch soil. Rotate crops every 3 years. Stake plants."
    },
    "Tomato___Late_blight": {
        "natural": "Bordeaux mixture. Baking soda spray. Remove ALL infected plants immediately.",
        "chemical": "Apply Mancozeb or Chlorothalonil URGENTLY. Do not compost infected material.",
        "prevention": "Resistant varieties. Avoid overhead watering. Monitor in cool wet weather."
    },
    "Tomato___Leaf_Mold": {
        "natural": "Neem oil spray. Improve ventilation. Reduce humidity below 85%.",
        "chemical": "Apply Chlorothalonil fungicide.",
        "prevention": "Maintain humidity below 85%. Improve greenhouse ventilation."
    },
    "Tomato___Septoria_leaf_spot": {
        "natural": "Compost tea spray. Copper soap. Remove infected leaves immediately.",
        "chemical": "Apply Mancozeb 75% WP at 7-10 day intervals.",
        "prevention": "Remove infected debris. Avoid wetting foliage. Stake plants."
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
        "natural": "No cure for virus. Remove infected plants. Apply neem oil to control whitefly vector.",
        "chemical": "Control whitefly with systemic insecticide. Remove infected plants.",
        "prevention": "Use virus-resistant varieties. Use reflective mulches to repel whitefly."
    },
    "Tomato___Tomato_mosaic_virus": {
        "natural": "No cure for virus. Remove and burn infected plants. Wash hands before handling plants.",
        "chemical": "No chemical cure. Remove infected plants.",
        "prevention": "Disinfect tools with 10% bleach. Use resistant varieties."
    },
    "Tomato___healthy": {
        "natural": "No treatment needed. Preventive neem oil spray monthly.",
        "chemical": "No treatment needed.",
        "prevention": "Regular monitoring. Proper nutrition and irrigation."
    },
}

# ═══════════════════════════════════════
# DATA TRANSFORMS
# ═══════════════════════════════════════
train_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ═══════════════════════════════════════
# LOAD DATASET
# ═══════════════════════════════════════
print(f"\n📂 Loading dataset from: {CONFIG['data_dir']}")
if not os.path.exists(CONFIG['data_dir']):
    print(f"❌ Dataset not found at {CONFIG['data_dir']}")
    print("Please update CONFIG['data_dir'] to your PlantVillage dataset path")
    exit(1)

full_dataset = datasets.ImageFolder(CONFIG['data_dir'], transform=train_transform)
n = len(full_dataset)
val_size = int(n * CONFIG['val_split'])
train_size = n - val_size
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_set.dataset.transform = val_transform

train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'],
                          shuffle=True, num_workers=CONFIG['num_workers'])
val_loader   = DataLoader(val_set, batch_size=CONFIG['batch_size'],
                          shuffle=False, num_workers=CONFIG['num_workers'])

classes = full_dataset.classes
print(f"✅ Dataset: {n} images, {len(classes)} classes")

# Save classes
os.makedirs("../model", exist_ok=True)
with open(CONFIG['classes_save_path'], 'w') as f:
    json.dump(classes, f)
print(f"✅ Classes saved to {CONFIG['classes_save_path']}")

# ═══════════════════════════════════════
# BUILD MODEL
# ═══════════════════════════════════════
print(f"\n🔧 Building MobileNetV2 for {len(classes)} classes...")
device = torch.device(CONFIG['device'])
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model = model.to(device)
print("✅ Model ready")

# ═══════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

best_acc = 0.0
print(f"\n🚀 Starting training for {CONFIG['num_epochs']} epochs...")

for epoch in range(CONFIG['num_epochs']):
    start = time.time()
    # Training
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()
        train_total += labels.size(0)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    train_acc = 100 * train_correct / train_total
    val_acc   = 100 * val_correct / val_total
    elapsed   = time.time() - start
    scheduler.step()

    print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] "
          f"Time: {elapsed:.1f}s | "
          f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), CONFIG['model_save_path'])
        print(f"  ✅ Best model saved! Val Acc: {best_acc:.2f}%")

print(f"\n🎉 Image Model Training Complete!")
print(f"🏆 Best Val Accuracy: {best_acc:.2f}%")
print(f"✅ Model saved to: {CONFIG['model_save_path']}")

# Save image model metadata
img_metadata = {
    'accuracy': round(best_acc, 2),
    'n_classes': len(classes),
    'classes': classes,
    'image_size': CONFIG['image_size'],
    'natural_cures_available': list(NATURAL_CURES.keys())
}
with open("../model/image_metadata.json", "w") as f:
    json.dump(img_metadata, f, indent=2)

# Save natural cures
with open("../data/natural_cures.json", "w") as f:
    json.dump(NATURAL_CURES, f, indent=2)
print("✅ Natural cures database saved")
