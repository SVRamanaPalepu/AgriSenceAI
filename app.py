from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import json
import pickle
import numpy as np
import pandas as pd

# Load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import database
import assistant

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "opticrop-v4-assistant-secret-key")

# Base directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize database
database.init_db()

# Load the trained model
model_path = os.path.join(BASE_DIR, "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Load dataset and precompute crop-specific statistics for field conditions
data_path = os.path.join(BASE_DIR, "dataset", "Crop_recommendation.csv")
crop_stats = {}

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    for crop_name, grp in df.groupby("label"):
        crop_stats[crop_name] = {}
        for f in features:
            q25 = float(grp[f].quantile(0.25))
            q75 = float(grp[f].quantile(0.75))
            crop_stats[crop_name][f] = {
                "min": round(float(grp[f].min()), 2),
                "max": round(float(grp[f].max()), 2),
                "q25": round(q25, 2),
                "q75": round(q75, 2),
                "mean": round(float(grp[f].mean()), 2)
            }

# Load comparison results if available
comparison_data = {}
comparison_json_path = os.path.join(BASE_DIR, "static", "model_comparison.json")
if os.path.exists(comparison_json_path):
    with open(comparison_json_path, "r", encoding="utf-8") as f:
        comparison_data = json.load(f)

def analyze_soil(crop_name, user_inputs):
    soil_params = [
        ("N", "Nitrogen (N)", "kg/ha"),
        ("P", "Phosphorus (P)", "kg/ha"),
        ("K", "Potassium (K)", "kg/ha"),
        ("ph", "Soil pH", "")
    ]
    analysis = []
    stats = crop_stats.get(crop_name.lower(), {})
    crop_display = crop_name.capitalize()
    
    for key, display_name, unit in soil_params:
        val = user_inputs.get(key, 0.0)
        c_stat = stats.get(key, None)
        
        if c_stat:
            q25 = c_stat["q25"]
            q75 = c_stat["q75"]
            c_min = c_stat["min"]
            c_max = c_stat["max"]
            mean_val = c_stat["mean"]
            
            if q25 <= val <= q75:
                status_text = "Suitable"
                status_icon = "✅"
                badge_class = "suitable"
                explanation = f"Your {display_name.lower()} level ({val} {unit}) is within the optimal central range ({q25} - {q75} {unit}) observed for {crop_display} in the dataset."
            elif c_min <= val <= c_max:
                status_text = "Moderate"
                status_icon = "⚠️"
                badge_class = "moderate"
                explanation = f"Your {display_name.lower()} level ({val} {unit}) falls within the acceptable historical limits ({c_min} - {c_max} {unit}) for {crop_display}."
            else:
                status_text = "Needs Attention"
                status_icon = "❗"
                badge_class = "attention"
                if val < c_min:
                    explanation = f"Your {display_name.lower()} level ({val} {unit}) is lower than the typical dataset range for {crop_display} (average: {mean_val} {unit})."
                else:
                    explanation = f"Your {display_name.lower()} level ({val} {unit}) is higher than the typical dataset range for {crop_display} (average: {mean_val} {unit})."
        else:
            status_text = "Suitable"
            status_icon = "✅"
            badge_class = "suitable"
            mean_val = val
            explanation = f"Your {display_name.lower()} level appears aligned with general agricultural patterns."
            
        analysis.append({
            "key": key,
            "name": display_name,
            "value": val,
            "unit": unit,
            "crop_avg": mean_val,
            "status_text": status_text,
            "status_icon": status_icon,
            "badge_class": badge_class,
            "explanation": explanation
        })
    return analysis

def analyze_climate(crop_name, user_inputs):
    climate_params = [
        ("temperature", "Temperature", "°C"),
        ("humidity", "Relative Humidity", "%"),
        ("rainfall", "Rainfall", "mm")
    ]
    analysis = []
    stats = crop_stats.get(crop_name.lower(), {})
    crop_display = crop_name.capitalize()
    
    for key, display_name, unit in climate_params:
        val = user_inputs.get(key, 0.0)
        c_stat = stats.get(key, None)
        
        if c_stat:
            q25 = c_stat["q25"]
            q75 = c_stat["q75"]
            c_min = c_stat["min"]
            c_max = c_stat["max"]
            mean_val = c_stat["mean"]
            
            if q25 <= val <= q75:
                status_text = "Suitable"
                status_icon = "✅"
                badge_class = "suitable"
                explanation = f"Your {display_name.lower()} ({val} {unit}) is well-suited, falling in the middle 50% range ({q25} - {q75} {unit}) for {crop_display}."
            elif c_min <= val <= c_max:
                status_text = "Moderate"
                status_icon = "⚠️"
                badge_class = "moderate"
                explanation = f"Your {display_name.lower()} ({val} {unit}) is within acceptable seasonal limits ({c_min} - {c_max} {unit}) for {crop_display}."
            else:
                status_text = "Needs Attention"
                status_icon = "❗"
                badge_class = "attention"
                if val < c_min:
                    explanation = f"Your {display_name.lower()} ({val} {unit}) is below the typical training baseline for {crop_display} (typical average: {mean_val} {unit})."
                else:
                    explanation = f"Your {display_name.lower()} ({val} {unit}) exceeds the typical training baseline for {crop_display} (typical average: {mean_val} {unit})."
        else:
            status_text = "Suitable"
            status_icon = "✅"
            badge_class = "suitable"
            mean_val = val
            explanation = f"Your {display_name.lower()} matches standard environmental patterns."
            
        analysis.append({
            "key": key,
            "name": display_name,
            "value": val,
            "unit": unit,
            "crop_avg": mean_val,
            "status_text": status_text,
            "status_icon": status_icon,
            "badge_class": badge_class,
            "explanation": explanation
        })
    return analysis

def get_overall_field_summary(crop_name, soil_analysis, climate_analysis):
    soil_suitable = sum(1 for s in soil_analysis if s["status_text"] == "Suitable")
    soil_attention = sum(1 for s in soil_analysis if s["status_text"] == "Needs Attention")
    
    if soil_attention == 0 and soil_suitable >= 3:
        soil_status = "Good"
        soil_badge = "suitable"
    elif soil_attention <= 1:
        soil_status = "Moderate"
        soil_badge = "moderate"
    else:
        soil_status = "Needs Attention"
        soil_badge = "attention"

    climate_suitable = sum(1 for c in climate_analysis if c["status_text"] == "Suitable")
    climate_attention = sum(1 for c in climate_analysis if c["status_text"] == "Needs Attention")
    
    if climate_attention == 0 and climate_suitable >= 2:
        climate_status = "Suitable"
        climate_badge = "suitable"
    elif climate_attention <= 1:
        climate_status = "Moderate"
        climate_badge = "moderate"
    else:
        climate_status = "Needs Attention"
        climate_badge = "attention"

    crop_display = crop_name.capitalize()
    summary_text = (
        f"The entered conditions are consistent with conditions associated with "
        f"{crop_display} in the training dataset."
    )
    return {
        "soil_status": soil_status,
        "soil_badge": soil_badge,
        "climate_status": climate_status,
        "climate_badge": climate_badge,
        "recommended_crop": crop_display,
        "summary_text": summary_text
    }

def generate_simple_explanation(crop_name, user_inputs, soil_analysis, climate_analysis):
    crop_display = crop_name.capitalize()
    return (
        f"Your soil nutrient levels (Nitrogen: {user_inputs['N']}, Phosphorus: {user_inputs['P']}, "
        f"Potassium: {user_inputs['K']}), soil pH ({user_inputs['ph']}), and climate conditions "
        f"(Temperature: {user_inputs['temperature']}°C, Humidity: {user_inputs['humidity']}%, "
        f"Rainfall: {user_inputs['rainfall']} mm) closely match conditions associated with "
        f"{crop_display} in the training dataset."
    )

def resolve_active_prediction(req_id=None):
    if req_id:
        pred = database.get_prediction_by_id(int(req_id))
        if pred:
            return pred
    session_id = session.get("last_pred_id")
    if session_id:
        pred = database.get_prediction_by_id(session_id)
        if pred:
            return pred
    all_preds = database.get_all_predictions()
    if all_preds:
        return all_preds[0]
    return None

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# About page
@app.route("/about")
def about():
    return render_template("about.html")

# Find Crop input form page
@app.route("/findcrop")
def findcrop():
    return render_template("index.html")

# Model Performance & Technical Evaluation page
@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html", comparison=comparison_data)

# Prediction History page
@app.route("/history")
def history():
    predictions_list = database.get_all_predictions()
    return render_template("history.html", predictions=predictions_list)

# Prediction Detail page
@app.route("/history/<int:pred_id>")
def history_detail(pred_id):
    pred = database.get_prediction_by_id(pred_id)
    if not pred:
        return redirect(url_for("history"))
    return render_template("history_detail.html", pred=pred)

# Clear History endpoint
@app.route("/history/clear", methods=["POST"])
def history_clear():
    database.clear_all_predictions()
    session.pop("last_pred_id", None)
    return redirect(url_for("history"))

# Page 1 - Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        user_inputs = {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        sample = pd.DataFrame({
            "N": [N], "P": [P], "K": [K],
            "temperature": [temperature],
            "humidity": [humidity],
            "ph": [ph],
            "rainfall": [rainfall]
        })

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            classes = model.classes_
            top3_indices = np.argsort(probabilities)[::-1][:3]
            
            top_predictions = []
            for rank, idx in enumerate(top3_indices, 1):
                prob_pct = round(float(probabilities[idx]) * 100, 1)
                top_predictions.append({
                    "rank": rank,
                    "crop": classes[idx],
                    "confidence": prob_pct
                })
            primary_crop = top_predictions[0]["crop"]
            confidence = top_predictions[0]["confidence"]
        else:
            primary_crop = model.predict(sample)[0]
            confidence = 100.0
            top_predictions = [{"rank": 1, "crop": primary_crop, "confidence": 100.0}]

        soil_analysis = analyze_soil(primary_crop, user_inputs)
        climate_analysis = analyze_climate(primary_crop, user_inputs)
        field_summary = get_overall_field_summary(primary_crop, soil_analysis, climate_analysis)
        explanation = generate_simple_explanation(primary_crop, user_inputs, soil_analysis, climate_analysis)

        pred_id = database.save_prediction(
            user_inputs=user_inputs,
            primary_crop=primary_crop,
            confidence=confidence,
            top_predictions=top_predictions,
            soil_analysis=soil_analysis,
            climate_analysis=climate_analysis,
            field_summary=field_summary,
            explanation=explanation
        )
        session["last_pred_id"] = pred_id

        return render_template(
            "result.html",
            prediction=primary_crop,
            confidence=confidence,
            top_predictions=top_predictions,
            field_summary=field_summary,
            explanation=explanation,
            inputs=user_inputs,
            pred_id=pred_id
        )
    except Exception as e:
        return render_template("result.html", error=str(e))

# Page 2 - Soil & Climate Analysis Route
@app.route("/analysis")
def analysis():
    req_id = request.args.get("id")
    pred = resolve_active_prediction(req_id)
    if not pred:
        return redirect(url_for("findcrop"))
    return render_template("analysis.html", pred=pred)

# Page 3 - Field Visual Insights Route
@app.route("/insights")
def insights():
    req_id = request.args.get("id")
    pred = resolve_active_prediction(req_id)
    if not pred:
        return redirect(url_for("findcrop"))
    return render_template("insights.html", pred=pred)

# ===================================================
# V4: AI Agricultural Assistant Endpoints
# ===================================================

@app.route("/assistant")
def assistant_page():
    req_id = request.args.get("id")
    pred = resolve_active_prediction(req_id)
    chat_history = session.get("chat_history", [])
    return render_template("assistant.html", pred=pred, chat_history=chat_history)

@app.route("/api/assistant/chat", methods=["POST"])
def assistant_chat():
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        
        if not message:
            return jsonify({"response": "Please enter an agricultural question."}), 400
            
        req_id = data.get("id")
        pred = resolve_active_prediction(req_id)
        chat_history = session.get("chat_history", [])
        
        response_text = assistant.get_assistant_response(message, context=pred, chat_history=chat_history)
        
        # Update session chat history
        chat_history.append({"sender": "user", "text": message})
        chat_history.append({"sender": "assistant", "text": response_text})
        session["chat_history"] = chat_history[-20:]  # keep recent 20 messages
        
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({
            "response": "Sorry, the Agricultural Assistant is temporarily unavailable. Please try again."
        }), 500

@app.route("/api/assistant/clear", methods=["POST"])
def assistant_clear():
    session["chat_history"] = []
    return jsonify({"status": "cleared"})

# Run the application
if __name__ == "__main__":
    app.run(debug=True)
