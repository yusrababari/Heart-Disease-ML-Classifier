"""Convert heart_model.pkl (LGBMClassifier) to ONNX format.

Run once locally before deploying:
    python export_model_onnx.py

The ONNX model can be loaded with onnxruntime (~19 MB, no scipy/sklearn/numpy deps),
cutting the Vercel bundle to ~50 MB total.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parent
PKL_PATH = ROOT / "heart_model.pkl"
OUT_PATH = ROOT / "api" / "heart_model.onnx"
FEATURE_NAMES_PATH = ROOT / "feature_names.json"

with open(FEATURE_NAMES_PATH) as f:
    feature_names = json.load(f)

n_features = len(feature_names)
print(f"Features ({n_features}): {feature_names}")

model = joblib.load(PKL_PATH)

initial_type = [("float_input", FloatTensorType([None, n_features]))]
onnx_model = convert_lightgbm(model, initial_types=initial_type, zipmap=False)

OUT_PATH.write_bytes(onnx_model.SerializeToString())
print(f"✓  Saved ONNX model → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")

# Quick sanity check
import onnxruntime as rt
sess = rt.InferenceSession(str(OUT_PATH))
dummy = np.zeros((1, n_features), dtype=np.float32)
label, proba = sess.run(None, {"float_input": dummy})
print(f"✓  Sanity check: label={label[0]}  proba={proba[0][1]:.4f}")
