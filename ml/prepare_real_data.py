"""
AgriCure 2.0 — Real Data Preparation Script
Run this AFTER downloading real dataset from Kaggle/ICRISAT
"""
import pandas as pd
import numpy as np
import json, os

print("🌱 AgriCure 2.0 — Real Data Preparation")
print("=" * 50)

os.makedirs("../data", exist_ok=True)

# ══════════════════════════════════════
# STEP 1 — Open your CSV in Excel first
# Check column names, then update below
# ══════════════════════════════════════
def standardize_columns(df):
    # ⚠️ CHANGE LEFT SIDE TO MATCH YOUR CSV COLUMNS
    rename_map = {
        'Temp_C'        : 'temperature',
        'RH_percent'    : 'humidity',
        'Rainfall_mm'   : 'rainfall',
        'Wind_kmh'      : 'wind_speed',
        'Month'         : 'month',
        'District'      : 'region',
        'State'         : 'state',
        'Crop'          : 'crop',
        'Disease_Name'  : 'disease',
        'Latitude'      : 'latitude',
        'Longitude'     : 'longitude',
    }
    df = df.rename(columns=rename_map)
    print(f"Columns after rename: {list(df.columns)}")
    return df

def add_missing_columns(df):
    if 'pesticide_applications' not in df.columns:
        df['pesticide_applications'] = 0
    if 'past_disease_occurrences' not in df.columns:
        df['past_disease_occurrences'] = 0
    if 'latitude' not in df.columns:
        df['latitude'] = 20.93
    if 'longitude' not in df.columns:
        df['longitude'] = 77.75
    if 'state' not in df.columns:
        df['state'] = 'Maharashtra'
    return df

DISEASE_NAME_MAP = {
    'late blight'           : 'Tomato___Late_blight',
    'Late Blight'           : 'Tomato___Late_blight',
    'tomato late blight'    : 'Tomato___Late_blight',
    'early blight'          : 'Tomato___Early_blight',
    'Early Blight'          : 'Tomato___Early_blight',
    'leaf rust'             : 'Wheat___Leaf_rust',
    'Leaf Rust'             : 'Wheat___Leaf_rust',
    'wheat rust'            : 'Wheat___Leaf_rust',
    'common rust'           : 'Corn_(maize)___Common_rust_',
    'northern leaf blight'  : 'Corn_(maize)___Northern_Leaf_Blight',
    'black rot'             : 'Apple___Black_rot',
    'apple scab'            : 'Apple___Apple_scab',
    'bacterial spot'        : 'Tomato___Bacterial_spot',
    'potato blight'         : 'Potato___Late_blight',
    'healthy'               : 'No_disease',
    'no disease'            : 'No_disease',
    'Healthy'               : 'No_disease',
}

def standardize_disease_names(df):
    df['disease'] = df['disease'].str.strip()
    df['disease'] = df['disease'].replace(DISEASE_NAME_MAP)
    return df

def clean_data(df):
    required = ['temperature', 'humidity', 'crop', 'disease']
    df = df.dropna(subset=required)
    df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    df['humidity']    = pd.to_numeric(df['humidity'], errors='coerce')
    df['rainfall']    = pd.to_numeric(df['rainfall'], errors='coerce').fillna(0)
    df['wind_speed']  = pd.to_numeric(df['wind_speed'], errors='coerce').fillna(5)
    df['month']       = pd.to_numeric(df['month'], errors='coerce').fillna(6).astype(int)
    df = df[df['temperature'].between(-10, 60)]
    df = df[df['humidity'].between(0, 100)]
    df = df[df['rainfall'] >= 0]
    df = df.drop_duplicates()
    print(f"✅ After cleaning: {len(df)} samples")
    return df

def add_features(df):
    df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
    df['is_monsoon']  = df['month'].apply(lambda m: 1 if 6 <= m <= 9 else 0)
    df['is_winter']   = df['month'].apply(lambda m: 1 if m <= 2 or m >= 11 else 0)
    df['is_summer']   = df['month'].apply(lambda m: 1 if 3 <= m <= 5 else 0)
    df['high_humidity'] = (df['humidity'] > 75).astype(int)
    df['high_rainfall'] = (df['rainfall'] > 20).astype(int)
    df['pesticide_effectiveness'] = np.clip(df['pesticide_applications'] * 0.2, 0, 1)
    return df

def prepare_data(input_file):
    print(f"\n📂 Loading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    df = standardize_columns(df)
    df = add_missing_columns(df)
    df = standardize_disease_names(df)
    df = clean_data(df)
    df = add_features(df)

    output = "../data/disease_prediction_dataset.csv"
    df.to_csv(output, index=False)
    print(f"\n✅ Saved to: {output}")
    print(f"Total: {len(df)} samples")
    print(f"\nDisease distribution:\n{df['disease'].value_counts()}")
    return df

if __name__ == "__main__":
    # ⚠️ CHANGE THIS to your downloaded file path
    INPUT_FILE = "../data/raw/real_disease_dataset.csv"

    if not os.path.exists(INPUT_FILE):
        print(f"❌ File not found: {INPUT_FILE}")
        print("\nDownload real dataset from:")
        print("1. Kaggle:  kaggle.com/datasets?search=crop+disease+india+weather")
        print("2. ICRISAT: data.icrisat.org")
        print("3. IMD:     imdpune.gov.in")
        print(f"\nSave it to: {INPUT_FILE}")
        print("\nThen open the CSV in Excel to check column names")
        print("and update the rename_map in standardize_columns()")
    else:
        prepare_data(INPUT_FILE)
