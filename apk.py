from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    result_class = None
    
    if request.method == "POST":
        age = float(request.form["age"])
        systolic = float(request.form["systolic"])
        diastolic = float(request.form["diastolic"])
        bmi = float(request.form["bmi"])
        cholesterol = float(request.form["cholesterol"])
        blood_sugar = float(request.form["blood_sugar"])
        heart_rate = float(request.form["heart_rate"])
        smoking = int(request.form["smoking"])
        alcohol = int(request.form["alcohol"])
        exercise = int(request.form["exercise"])
        family_history = int(request.form["family_history"])
        
        pulse_pressure = systolic - diastolic
        map_val = diastolic + (pulse_pressure / 3)
        
        if age <= 35: age_group = 2
        elif age <= 60: age_group = 1
        else: age_group = 0
        
        features = np.array([[age, systolic, diastolic, bmi,
                              cholesterol, blood_sugar, heart_rate,
                              pulse_pressure, map_val, smoking,
                              alcohol, exercise, family_history,
                              age_group]])
        
        result = model.predict(features)[0]
        
        if result == 0:
            prediction = "🔴 HIGH RISK"
            result_class = "high-risk"
        else:
            prediction = "🟢 LOW RISK"
            result_class = "low-risk"
    
    return render_template("index.html", 
                         prediction=prediction,
                         result_class=result_class)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
