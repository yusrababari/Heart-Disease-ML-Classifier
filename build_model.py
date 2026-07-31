"""Enrich the heart disease dataset with weight / smoking / diabetes factors,
retrain an LGBM classifier using the same feature pipeline as the notebook,
and export the model + feature names consumed by main.py.
"""

import json
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parent
FULL_CSV = ROOT / "heart_disease_full.csv"
TRAIN_CSV = ROOT / "data" / "train.csv"
TEST_CSV = ROOT / "data" / "test.csv"
MODEL_PATH = ROOT / "heart_model.pkl"
FEATURES_PATH = ROOT / "feature_names.json"

CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]
RNG_SEED = 42


def add_synthetic_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic `weight`, `smoking` and `diabetes` columns."""
    rng = np.random.default_rng(RNG_SEED)
    n = len(df)

    sex = df["sex"].to_numpy(dtype=float)
    age = df["age"].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    fbs = df["fbs"].to_numpy(dtype=float)

    # Weight (kg): men heavier than women, slight increase with age,
    # and a mild positive association with heart disease risk.
    base = np.where(sex == 1, 78.0, 66.0) + (age - 50) * 0.08
    weight = base + target * 3.0 + rng.normal(0, 11.5, n)
    df["weight"] = np.clip(weight.round(1), 40.0, 140.0)

    # Smoking: more common in males and among heart disease patients.
    smoke_prob = np.where(sex == 1, 0.40, 0.22) + target * 0.06
    df["smoking"] = (rng.uniform(0, 1, n) < smoke_prob).astype(int)

    # Diabetes: elevated fasting blood sugar strongly raises the odds,
    # with an additional lift when heart disease is present.
    diab_prob = 0.14 + fbs * 0.45 + target * 0.10
    df["diabetes"] = (rng.uniform(0, 1, n) < np.clip(diab_prob, 0.0, 0.95)).astype(int)

    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the notebook's feature engineering."""
    df = df.copy()
    df["AgeGroup"] = pd.cut(df["age"], bins=[0, 40, 55, 100], labels=[0, 1, 2]).astype(int)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df


def main() -> None:
    full = pd.read_csv(FULL_CSV)
    full = add_synthetic_factors(full)
    full["ca"] = full["ca"].fillna(0.0)
    full["thal"] = full["thal"].fillna(2.0)

    train, test = train_test_split(
        full, test_size=0.2, random_state=RNG_SEED, stratify=full["target"]
    )
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    train_fe = feature_engineer(train)
    test_fe = feature_engineer(test)
    train_fe, test_fe = train_fe.align(test_fe, join="left", axis=1, fill_value=0)
    if "target" in test_fe.columns:
        test_fe["target"] = test_fe["target"].fillna(0).astype(int)

    feature_names = list(train_fe.drop(columns=["target"]).columns)
    X_train, y_train = train_fe[feature_names], train_fe["target"]
    X_test, y_test = test_fe[feature_names], test_fe["target"]

    model = lightgbm.LGBMClassifier(random_state=RNG_SEED, verbosity=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Train shape: {X_train.shape[0]}, Test shape: {X_test.shape[0]}")
    print(f"Feature count: {len(feature_names)}")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (std = {cv_scores.std():.4f})")
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["No Disease", "Disease"]))

    joblib.dump(model, MODEL_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Exported {MODEL_PATH.name} and {FEATURES_PATH.name}")


if __name__ == "__main__":
    main()
