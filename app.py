from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat

MODEL_PATH = Path("model.joblib")

FEATURE_NAMES = [
    "age",
    "sex",
    "chest",
    "resting_blood_pressure",
    "serum_cholestoral",
    "fasting_blood_sugar",
    "resting_electrocardiographic_results",
    "maximum_heart_rate_achieved",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "number_of_major_vessels",
    "thal",
]


class HeartDiseaseFeatures(BaseModel):
    age: confloat(ge=0, le=120) = Field(..., description="Age in years")
    sex: conint(ge=0, le=1) = Field(..., description="1 = male, 0 = female")
    chest: conint(ge=1, le=4) = Field(..., description="Chest pain type (1-4)")
    resting_blood_pressure: confloat(ge=1) = Field(..., description="Resting blood pressure")
    serum_cholestoral: confloat(ge=1) = Field(..., description="Serum cholesterol in mg/dl")
    fasting_blood_sugar: conint(ge=0, le=1) = Field(..., description="Fasting blood sugar > 120 mg/dl")
    resting_electrocardiographic_results: conint(ge=0, le=2) = Field(..., description="Resting electrocardiographic results")
    maximum_heart_rate_achieved: confloat(ge=1) = Field(..., description="Maximum heart rate achieved")
    exercise_induced_angina: conint(ge=0, le=1) = Field(..., description="Exercise induced angina")
    oldpeak: confloat(ge=0) = Field(..., description="ST depression induced by exercise relative to rest")
    slope: conint(ge=1, le=3) = Field(..., description="Slope of the peak exercise ST segment")
    number_of_major_vessels: conint(ge=0, le=3) = Field(..., description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: conint(ge=3, le=7) = Field(..., description="Thalassemia code, usually 3, 6, or 7")


class PredictionResponse(BaseModel):
    prediction: str
    probability_present: float
    probability_absent: float
    explanation: str


app = FastAPI(
    title="Heart Disease Classifier",
    description="FastAPI service wrapping a trained heart disease model for binary prediction.",
    version="1.0.0",
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find model file at {MODEL_PATH}. Run train.py to build model.joblib."
        )
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_names"]


model, model_feature_names = load_model()


def normalize_features(features: HeartDiseaseFeatures) -> pd.DataFrame:
    return pd.DataFrame(
        [{name: getattr(features, name) for name in model_feature_names}],
        columns=model_feature_names,
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartDiseaseFeatures):
    inputs = normalize_features(features)
    try:
        probabilities = model.predict_proba(inputs)[0]
        predicted = model.predict(inputs)[0]
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="The trained model does not support probability predictions.",
        )

    label = "present" if int(predicted) == 1 else "absent"
    probability_present = float(probabilities[1])
    probability_absent = float(probabilities[0])
    explanation = (
        f"Predicted heart disease is {label} with {probability_present:.2f} probability."
    )

    return {
        "prediction": label,
        "probability_present": round(probability_present, 4),
        "probability_absent": round(probability_absent, 4),
        "explanation": explanation,
    }


@app.post("/batch_predict", response_model=List[PredictionResponse])
def batch_predict(samples: List[HeartDiseaseFeatures]):
    if not samples:
        raise HTTPException(status_code=400, detail="Submit at least one sample for batch prediction.")

    input_values = pd.DataFrame(
        [{name: getattr(sample, name) for name in model_feature_names} for sample in samples],
        columns=model_feature_names,
    )
    try:
        probabilities = model.predict_proba(input_values)
        predicted = model.predict(input_values)
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="The trained model does not support probability predictions.",
        )

    results = []
    for proba, label in zip(probabilities, predicted):
        label_text = "present" if int(label) == 1 else "absent"
        results.append(
            {
                "prediction": label_text,
                "probability_present": round(float(proba[1]), 4),
                "probability_absent": round(float(proba[0]), 4),
                "explanation": (
                    f"Predicted heart disease is {label_text} with {float(proba[1]):.2f} probability."
                ),
            }
        )
    return results


@app.get("/")
def root():
    return {
        "service": "Heart Disease Classifier",
        "description": "Send patient features to /predict to get a heart disease probability.",
        "endpoints": ["/predict", "/batch_predict", "/health"],
    }
