import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graphs import generate_graphs

ROOT = Path(__file__).resolve().parent

app = FastAPI()

# Allow the React frontend to connect (same-origin on Vercel, cross-origin elsewhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved assets
model = joblib.load(ROOT / "heart_model.pkl")
with open(ROOT / "feature_names.json", "r") as f:
    feature_names = json.load(f)


class HeartData(BaseModel):
    features: dict  # Example: {"age": 63, "sex": 1, "cp": 3, ...}

@app.get("/api")
def read_root():
    return {"message": "Heart Disease ML Classifier API is running!"}

@app.get("/api/features")
def get_features():
    """Returns required feature names to build dynamic UI inputs."""
    return {"features": feature_names}

@app.get("/api/graphs")
def get_graphs():
    """Regenerates all notebook visualizations and returns them as data URIs."""
    return {"graphs": generate_graphs()}

@app.post("/api/predict")
def predict(data: HeartData):
    """Accepts user feature inputs and returns model prediction."""
    # Format features into DataFrame with exact column order
    df = pd.DataFrame([data.features])[feature_names]
    prediction = model.predict(df)[0]

    # If your model supports predict_proba
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": int(prediction),
        "probability": probability
    }
