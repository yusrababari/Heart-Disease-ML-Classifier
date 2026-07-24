from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report

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
CATEGORICAL_FEATURES = [
    "chest",
    "resting_electrocardiographic_results",
    "slope",
    "thal",
]
NUMERIC_FEATURES = [name for name in FEATURE_NAMES if name not in CATEGORICAL_FEATURES]


def load_heart_dataset():
    dataset = fetch_openml(data_id=53, as_frame=True)
    X = dataset.data[FEATURE_NAMES]
    y = dataset.target.astype(str).replace({"absent": 0, "present": 1})
    y = pd.to_numeric(y, errors="raise").astype(int)
    return X, y


def build_pipeline():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=200, random_state=42),
            ),
        ]
    )


def train_and_save_model(model_path: Path = MODEL_PATH):
    X, y = load_heart_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["absent", "present"])
    print("Heart disease classifier training complete")
    print(report)

    joblib.dump({"model": pipeline, "feature_names": FEATURE_NAMES}, model_path)
    return pipeline


if __name__ == "__main__":
    train_and_save_model()
