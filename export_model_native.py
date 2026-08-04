"""Convert heart_model.pkl (LGBMClassifier joblib) to native LightGBM .txt format.

Run once locally before deploying:
    python export_model_native.py

The native model can be loaded with lgb.Booster(model_file=...) — no scikit-learn
or scipy needed at runtime, saving ~170 MB from the Vercel bundle.
"""

from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
PKL_PATH = ROOT / "heart_model.pkl"
OUT_PATH = ROOT / "api" / "heart_model.txt"

model = joblib.load(PKL_PATH)

# LGBMClassifier wraps an underlying Booster — extract it
if hasattr(model, "booster_"):
    booster = model.booster_
elif hasattr(model, "_Booster"):
    booster = model._Booster
else:
    # Already a raw Booster
    booster = model

booster.save_model(str(OUT_PATH))
print(f"✓  Saved native LightGBM model → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
