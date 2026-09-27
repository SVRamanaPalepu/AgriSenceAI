# database.py - SQLite Storage for OptiCrop Prediction History
import sqlite3
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date_str TEXT NOT NULL,
            time_str TEXT NOT NULL,
            nitrogen REAL NOT NULL,
            phosphorus REAL NOT NULL,
            potassium REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            ph REAL NOT NULL,
            rainfall REAL NOT NULL,
            primary_crop TEXT NOT NULL,
            confidence REAL NOT NULL,
            top_predictions_json TEXT NOT NULL,
            soil_analysis_json TEXT NOT NULL,
            climate_analysis_json TEXT NOT NULL,
            field_summary_json TEXT NOT NULL,
            explanation TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(user_inputs, primary_crop, confidence, top_predictions, soil_analysis, climate_analysis, field_summary, explanation):
    init_db()
    now = datetime.now()
    date_str = now.strftime("%d %b %Y")
    time_str = now.strftime("%I:%M %p")
    timestamp = now.isoformat()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, date_str, time_str,
            nitrogen, phosphorus, potassium,
            temperature, humidity, ph, rainfall,
            primary_crop, confidence,
            top_predictions_json, soil_analysis_json,
            climate_analysis_json, field_summary_json, explanation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, date_str, time_str,
        user_inputs["N"], user_inputs["P"], user_inputs["K"],
        user_inputs["temperature"], user_inputs["humidity"], user_inputs["ph"], user_inputs["rainfall"],
        primary_crop, confidence,
        json.dumps(top_predictions), json.dumps(soil_analysis),
        json.dumps(climate_analysis), json.dumps(field_summary), explanation
    ))
    pred_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return pred_id

def get_all_predictions():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    predictions = []
    for r in rows:
        predictions.append({
            "id": r["id"],
            "date_str": r["date_str"],
            "time_str": r["time_str"],
            "nitrogen": r["nitrogen"],
            "phosphorus": r["phosphorus"],
            "potassium": r["potassium"],
            "temperature": r["temperature"],
            "humidity": r["humidity"],
            "ph": r["ph"],
            "rainfall": r["rainfall"],
            "primary_crop": r["primary_crop"],
            "confidence": r["confidence"],
            "top_predictions": json.loads(r["top_predictions_json"]),
            "soil_analysis": json.loads(r["soil_analysis_json"]),
            "climate_analysis": json.loads(r["climate_analysis_json"]),
            "field_summary": json.loads(r["field_summary_json"]),
            "explanation": r["explanation"]
        })
    return predictions

def get_prediction_by_id(pred_id):
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
    r = cursor.fetchone()
    conn.close()

    if not r:
        return None

    return {
        "id": r["id"],
        "date_str": r["date_str"],
        "time_str": r["time_str"],
        "nitrogen": r["nitrogen"],
        "phosphorus": r["phosphorus"],
        "potassium": r["potassium"],
        "temperature": r["temperature"],
        "humidity": r["humidity"],
        "ph": r["ph"],
        "rainfall": r["rainfall"],
        "primary_crop": r["primary_crop"],
        "confidence": r["confidence"],
        "top_predictions": json.loads(r["top_predictions_json"]),
        "soil_analysis": json.loads(r["soil_analysis_json"]),
        "climate_analysis": json.loads(r["climate_analysis_json"]),
        "field_summary": json.loads(r["field_summary_json"]),
        "explanation": r["explanation"]
    }

def clear_all_predictions():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DB_PATH)
