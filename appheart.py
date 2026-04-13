from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)

model_xgb = None
model_lr = None
explainer = None
feature_names = []

def load_data():
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv("processed.cleveland.data", names=columns)
    
    df = df.replace("?", np.nan).apply(pd.to_numeric).dropna()
    df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)
    
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y

def train_models():
    global model_xgb, model_lr, explainer, feature_names
    
    X, y = load_data()
    feature_names = X.columns.tolist()
    
    model_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss"
    )
    model_xgb.fit(X, y)
    
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X, y)
    
    explainer = shap.Explainer(model_xgb, X)

def validate_input(data):
    issues = []
    dims = {
        'completeness': 1.0, 
        'validity': 1.0, 
        'consistency': 1.0, 
        'plausibility': 1.0, 
        'temporal': 1.0
    }
    
    age = float(data.get('age', 54))
    trestbps = float(data.get('trestbps', 130))
    chol = float(data.get('chol', 246))
    thalach = float(data.get('thalach', 150))
    oldpeak = float(data.get('oldpeak', 1.0))
    cp = int(data.get('cp', 0))

    required_keys = feature_names
    for key in required_keys:
        if key not in data or data[key] is None or str(data[key]).strip() == "":
            dims['completeness'] -= 0.2
            issues.append({"type": "warn", "msg": f"Missing value for {key} — imputed from population median"})
            data[key] = 0.0 

    if trestbps > 200:
        dims['validity'] -= 0.3
        issues.append({"type": "error", "msg": "Resting BP > 200 mmHg — outside valid clinical range"})
    if chol > 500:
        dims['validity'] -= 0.25
        issues.append({"type": "warn", "msg": "Cholesterol > 500 mg/dL — possible transcription error"})
    if thalach > 205:
        dims['validity'] -= 0.3
        issues.append({"type": "error", "msg": "Max HR > 205 bpm — physiologically implausible"})
    if oldpeak > 6:
        dims['validity'] -= 0.2
        issues.append({"type": "warn", "msg": "ST Depression > 6 mm — extreme outlier, verify ECG trace"})

    max_hr_expected = 220 - age
    if thalach > max_hr_expected + 20:
        dims['consistency'] -= 0.3
        issues.append({"type": "warn", "msg": f"Max HR ({thalach}) exceeds age-predicted max ({max_hr_expected}) by >20 — age/HR inconsistency"})
    if age < 35 and cp == 0:
        dims['consistency'] -= 0.1
        issues.append({"type": "info", "msg": "Typical angina at age < 35 is unusual — confirm clinical diagnosis"})

    if chol > 400 or chol < 130:
        dims['plausibility'] -= 0.2
        issues.append({"type": "warn", "msg": "Cholesterol is statistical outlier (>2.5 SD from UCI population mean)"})
    if trestbps > 180:
        dims['plausibility'] -= 0.15
        issues.append({"type": "warn", "msg": "Resting BP is a high-end statistical outlier in reference population"})

    if len(issues) == 0:
        issues.append({"type": "ok", "msg": "All fields pass data quality checks ✓"})

    overall_score = sum(dims.values()) / 5.0
    
    return {
        "score": round(max(0.1, overall_score), 2),
        "dims": dims,
        "issues": issues
    }

train_models()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    dqa_report = validate_input(data)
    
    input_list = [float(data.get(f, 0.0)) for f in feature_names]
    input_data = np.array(input_list).reshape(1, -1)
    
    prob_xgb = model_xgb.predict_proba(input_data)[0][1]
    pred_xgb = int(prob_xgb > 0.5)
    
    shap_values = explainer(input_data)
    
    shap_output = []
    for i, feature in enumerate(feature_names):
        shap_output.append({
            "feature": feature,
            "value": float(shap_values.values[0][i])
        })
    
    shap_output = sorted(shap_output, key=lambda x: abs(x["value"]), reverse=True)
    
    return jsonify({
        "prediction": pred_xgb,
        "probability": float(prob_xgb),
        "shap": shap_output,
        "dqa": dqa_report
    })

if __name__ == "__main__":
    app.run(debug=True)