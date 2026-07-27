import json
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved assets
model = joblib.load("heart_model.pkl")
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

class HeartData(BaseModel):
    features: dict  # Example: {"age": 63, "sex": 1, "cp": 3, ...}
@app.get("/")
def read_root():
    return {"message": "Heart Disease ML Classifier API is running!"}
@app.get("/features")
def get_features():
    """Returns required feature names to build dynamic UI inputs."""
    return {"features": feature_names}

@app.post("/predict")
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
