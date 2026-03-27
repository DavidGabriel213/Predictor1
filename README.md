# 🏥 Nigerian Patient Disease Risk Predictor

A complete end-to-end Machine Learning web application predicting 
cardiovascular disease risk in Nigerian patients.

🌐 **Live App:** https://disease-predictor-q60i.onrender.com

---

## 📋 Project Overview

Complete ML pipeline from raw messy data to deployed web application.
825 patient records, 16 columns, extensive data quality issues.

---

## 🛠️ Tech Stack

- **Python** — core language
- **Pandas + NumPy** — data cleaning & engineering
- **Scikit-learn** — ML models + GridSearchCV
- **Pickle** — model persistence
- **Flask** — web backend
- **HTML + CSS** — frontend
- **Render.com** — deployment
- **UptimeRobot** — 24/7 monitoring

---

## 🧹 Data Cleaning (10 Issues Fixed)

| Column | Problem | Solution |
|---|---|---|
| Age | Outliers ×10, negatives, nulls | IQR both bounds |
| Systolic_BP | '120/80' combined strings | String split on '/' |
| BMI | '26.5kg/m2' strings | Strip units |
| Cholesterol | Outliers ×10 | IQR clipping |
| BloodSugar | '95mg/dL' strings | Strip units |
| Smoking | 6 mixed formats | str.capitalize() + dict |
| Duplicates | 25 hidden rows | drop_duplicates() |

---

## 📐 Feature Engineering

| Feature | Formula |
|---|---|
| PulsePressure | Systolic - Diastolic |
| MAP | Diastolic + (PulsePressure / 3) |
| BMI_Class | pd.cut(BMI, 5 bins) |
| Age_Group | pd.cut(Age, 3 bins) |

---

## 🤖 Model Results

| Model | Accuracy |
|---|---|
| **Logistic Regression** | **84.38%** 🏆 |
| RF Tuned (GridSearchCV) | 79.37% |
| Random Forest | 76.88% |
| Decision Tree | 70.00% |

---

## 📁 Project Structure
├── app.py                 ← Flask backend
├── model.pkl              ← Saved ML model
├── requirements.txt       ← Dependencies
├── Procfile               ← Render config
├── templates/
│   └── index.html         ← Frontend
└── static/
└── style.css          ← Styling
---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python app.py
# Visit http://127.0.0.1:5000
First complete ML deployment — self-taught, built on Android phone during IT placement.
Part of my journey toward becoming a Senior ML/AI Engineer.
GitHub: github.com/DavidGabriel213
LinkedIn: linkedin.com/in/gabriel-david-ds
