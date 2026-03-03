import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "csv_data" / "processed_fraud_train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model_rf.joblib"
ENCODER_PATH = PROJECT_ROOT / "models" / "encoders.joblib"

FEATURES = ["category", "amt", "gender", "state", "merchant", "hour", "day_of_week"]
CATEGORICAL_COLS = ["category", "gender", "state", "merchant"]

def retrain():
    if not DATA_PATH.exists():
        print(f"Error: Data file {DATA_PATH} not found.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print("Preprocessing data...")
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    X = df[FEATURES]
    y = df["is_fraud"]
    
    print(f"Training RandomForest with {len(X)} samples...")
    # Using similar parameters to common fraud models
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print("Training complete.")
    
    # Save the model and encoders
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print(f"Saving encoders to {ENCODER_PATH}...")
    joblib.dump(encoders, ENCODER_PATH)
    
    print("✅ Model and encoders successfully re-trained and saved.")

if __name__ == "__main__":
    retrain()
