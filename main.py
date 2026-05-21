from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder=".")

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

FEATURE_COLS = [
    "study_hours_per_day",
    "phone_usage_hours",
    "social_media_hours",
    "youtube_hours",
    "gaming_hours",
    "breaks_per_day",
    "coffee_intake_mg",
    "assignments_completed",
    "attendance_percentage",
    "focus_score",
    "sleep_hours",
    "stress_level",
    "exercise_minutes",
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)

        data = {col: [float(body[col])] for col in FEATURE_COLS}
        df = pd.DataFrame(data)[FEATURE_COLS]

        prepared = pipeline.transform(df)
        score = float(model.predict(prepared)[0])
        score = round(max(0, min(100, score)), 2)

        if score >= 75:
            label = "High"
        elif score >= 50:
            label = "Medium"
        else:
            label = "Low"

        return jsonify({"score": score, "label": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)