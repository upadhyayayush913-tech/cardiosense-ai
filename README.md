# CardioSense AI 🫀

**Explainable AI-Based Clinical Decision Support System for Heart Disease Prediction with Data Quality Assessment**

This project demonstrates a high-level, research-oriented approach to predicting heart disease risk. It extends traditional ML models by incorporating **real-time Data Quality Assessment (DQA)** and **Explainable AI (XAI)** to create a fully transparent, clinical-grade decision support system.

## ✨ Features

- **Ensemble Machine Learning:** Uses XGBoost (primary) and Logistic Regression to robustly predict cardiovascular risk based on the UCI Heart Disease dataset.
- **Explainable AI (SHAP):** Unboxes the black-box model. It calculates SHAP (SHapley Additive exPlanations) values to explain exactly *why* a specific prediction was made, breaking down the feature contributions (e.g., how much did cholesterol increase the risk score?).
- **Real-Time Data Quality Assessment (DQA):** Before a prediction is made, the backend performs a rigorous 5-dimension quality check (Completeness, Validity, Consistency, Plausibility, and Temporal Stability). It dynamically flags clinically implausible values or inconsistencies (like physiological mismatches between age and max heart rate).
- **Advanced Clinical UI:** A premium frontend dashboard presenting predictions, SHAP visualisations, data quality indices, and plain-language clinical interpretations.

## 🚀 How to Run

### 1. Install Dependencies
Make sure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server
Run the Flask server which handles ML inference, SHAP calculation, and DQA validation:
```bash
python appheart.py
```
*The server will start at `http://127.0.0.1:5000/`*

### 3. Open the Frontend Dashboard
Simply double-click the `heart_risk_demo.html` file to open it in your web browser. 
No local server is required for the frontend.

## 🧪 Sample Inputs to Try

Use these values in the UI to see how the system reacts to different clinical scenarios:

**1. High-Risk Patient (Typical)**
- Age: 63, Sex: Male, Chest Pain: Asymptomatic, Resting BP: 145, Cholesterol: 233, Max HR: 150

**2. Low-Risk Patient (Healthy)**
- Age: 37, Sex: Female, Chest Pain: Non-Anginal, Resting BP: 120, Cholesterol: 180, Max HR: 187

**3. Noisy / Impossible Data (Triggers DQA Warnings)**
- Age: 25, Sex: Male, Resting BP: 210, Cholesterol: 564, Max HR: 210
*(Notice how the Data Quality score drops and issues are flagged.)*

## 🧠 Explainability in Viva / Defense

If asked to explain this system during a viva:

* **What does the ML Model do?** It's trained on historical UCI clinical data. XGBoost is used due to its high tabular data performance. It learns patterns between 13 clinical features and the presence of heart disease.
* **Why SHAP?** SHAP treats the model as a cooperative game where features are players. It distributes the prediction "payout" fairly among features. It allows doctors to see that, for example, "the patient's risk is 80%, and +20% of that is solely due to their high cholesterol."
* **Why DQA?** "Garbage in, garbage out." Clinical data is often noisy or misentered. The backend intercepts the data and runs rule-based clinical checks (e.g., checking if max HR exceeds `220 - age`). It passes this confidence score to the UI, so doctors know whether to trust the prediction.
