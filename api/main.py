import json
from pathlib import Path

import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
_sess = rt.InferenceSession(str(ROOT / "heart_model.onnx"))
_input_name = _sess.get_inputs()[0].name
with open(ROOT / "feature_names.json", "r") as f:
    feature_names = json.load(f)

# Load pre-generated graphs (produced by generate_graphs_static.py)
_graphs_path = ROOT / "graphs.json"
if _graphs_path.exists():
    with open(_graphs_path, "r", encoding="utf-8") as f:
        _cached_graphs = json.load(f)
else:
    _cached_graphs = []


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
    """Returns pre-generated notebook visualizations as data URIs."""
    if not _cached_graphs:
        raise HTTPException(status_code=503, detail="Graphs not available. Run generate_graphs_static.py locally and commit api/graphs.json.")
    return {"graphs": _cached_graphs}

@app.post("/api/predict")
def predict(data: HeartData):
    """Accepts user feature inputs and returns model prediction."""
    try:
        row = np.array([[data.features[name] for name in feature_names]], dtype=np.float32)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")

    labels, probas = _sess.run(None, {_input_name: row})
    prediction = int(labels[0])
    probability = float(probas[0][1])

    return {
        "prediction": prediction,
        "probability": probability
    }
