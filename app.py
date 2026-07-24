from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.joblib")

# Global variables for model, feature names, and startup error tracking
model: Optional[Any] = None
model_feature_names: List[str] = []
model_load_error: Optional[str] = None


class HeartDiseaseFeatures(BaseModel):
    age: Annotated[float, Field(ge=0, le=120, description="Age in years")]
    sex: Annotated[int, Field(ge=0, le=1, description="1 = male, 0 = female")]
    chest: Annotated[int, Field(ge=1, le=4, description="Chest pain type (1-4)")]
    resting_blood_pressure: Annotated[float, Field(ge=1, description="Resting blood pressure")]
    serum_cholestoral: Annotated[float, Field(ge=1, description="Serum cholesterol in mg/dl")]
    fasting_blood_sugar: Annotated[int, Field(ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")]
    resting_electrocardiographic_results: Annotated[int, Field(ge=0, le=2, description="Resting electrocardiographic results")]
    maximum_heart_rate_achieved: Annotated[float, Field(ge=1, description="Maximum heart rate achieved")]
    exercise_induced_angina: Annotated[int, Field(ge=0, le=1, description="Exercise induced angina")]
    oldpeak: Annotated[float, Field(ge=0, description="ST depression induced by exercise relative to rest")]
    slope: Annotated[int, Field(ge=1, le=3, description="Slope of the peak exercise ST segment")]
    number_of_major_vessels: Annotated[int, Field(ge=0, le=3, description="Number of major vessels (0-3) colored by fluoroscopy")]
    thal: Annotated[int, Field(ge=3, le=7, description="Thalassemia code, usually 3, 6, or 7")]


class PredictionResponse(BaseModel):
    prediction: str
    probability_present: float
    probability_absent: float
    explanation: str


def load_model() -> Tuple[Any, List[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find model file at {MODEL_PATH}. Run train.py to build model.joblib."
        )
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["feature_names"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_feature_names, model_load_error
    if MODEL_PATH.exists():
        try:
            model, model_feature_names = load_model()
        except Exception as e:
            model_load_error = str(e)
    else:
        model_load_error = f"Model file not found at {MODEL_PATH}"
    yield


app = FastAPI(
    title="Heart Disease Classifier",
    description="FastAPI service wrapping a trained heart disease model for binary prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


def extract_features_df(samples: List[HeartDiseaseFeatures]) -> pd.DataFrame:
    """Converts Pydantic input models to a DataFrame with correct feature ordering."""
    data = [sample.model_dump() for sample in samples]
    df = pd.DataFrame(data)
    if model_feature_names:
        df = df.reindex(columns=model_feature_names)
    return df


def get_class_probabilities(probabilities_row: Any, classes: Any) -> Tuple[float, float]:
    """Helper to safely assign probabilities for present (1) and absent (0)."""
    class_map = {cls: idx for idx, cls in enumerate(classes)}
    idx_present = class_map.get(1, 1)
    idx_absent = class_map.get(0, 0)
    
    prob_present = float(probabilities_row[idx_present])
    prob_absent = float(probabilities_row[idx_absent])
    return prob_present, prob_absent


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path_exists": MODEL_PATH.exists(),
        "error": model_load_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartDiseaseFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded on the server. Details: {model_load_error}",
        )

    inputs = extract_features_df([features])
    
    try:
        probabilities = model.predict_proba(inputs)[0]
        predicted = model.predict(inputs)[0]
        classes = getattr(model, "classes_", [0, 1])
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="The trained model does not support probability predictions.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    prob_present, prob_absent = get_class_probabilities(probabilities, classes)
    label = "present" if int(predicted) == 1 else "absent"
    explanation = (
        f"Predicted heart disease is {label} with {prob_present:.2f} probability."
    )

    return {
        "prediction": label,
        "probability_present": round(prob_present, 4),
        "probability_absent": round(prob_absent, 4),
        "explanation": explanation,
    }


@app.post("/batch_predict", response_model=List[PredictionResponse])
def batch_predict(samples: List[HeartDiseaseFeatures]):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded on the server. Details: {model_load_error}",
        )

    if not samples:
        raise HTTPException(
            status_code=400, 
            detail="Submit at least one sample for batch prediction."
        )

    input_values = extract_features_df(samples)
    
    try:
        probabilities = model.predict_proba(input_values)
        predicted = model.predict(input_values)
        classes = getattr(model, "classes_", [0, 1])
    except AttributeError:
        raise HTTPException(
            status_code=500,
            detail="The trained model does not support probability predictions.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    results = []
    for proba, label in zip(probabilities, predicted):
        prob_present, prob_absent = get_class_probabilities(proba, classes)
        label_text = "present" if int(label) == 1 else "absent"
        results.append(
            {
                "prediction": label_text,
                "probability_present": round(prob_present, 4),
                "probability_absent": round(prob_absent, 4),
                "explanation": (
                    f"Predicted heart disease is {label_text} with {prob_present:.2f} probability."
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