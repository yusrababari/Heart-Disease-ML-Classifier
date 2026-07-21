import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

train=pd.read_csv("data/train.csv")
X=train.drop(columns=["target"])
y=train["target"]
model=RandomForestClassifier(random_state=42)
model.fit(X,y)
joblib.dump(model,"model.pkl")
print("Saved model.pkl")
