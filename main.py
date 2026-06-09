# ============================================================
# NESYAN — Cognitive Decline FastAPI Service
# POST /predict → parse frontend JSON, run model, save to DB, return result
# GET /predict/{id} → fetch latest result for a patient from DB
# ============================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
import numpy as np
import pandas as pd
import joblib
import shap
import os
import sqlite3
import json
from datetime import datetime, timezone

# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = os.getenv("MODEL_PATH", "alzheimers_model.pkl")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found: {MODEL_PATH}. Run the notebook first.")

# ============================================================
# DATABASE SETUP (SQLite — zero config, works on Render free tier)
# ============================================================
DB_PATH = os.getenv("DB_PATH", "nesyan.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    probabilities TEXT NOT NULL, -- JSON string
    alert TEXT NOT NULL,
    predicted_at TEXT NOT NULL
    )
    """)
    con.commit()
    con.close()

init_db()

def save_result(patient_id: str, result: dict):
    con = sqlite3.connect(DB_PATH)
    con.execute("""
    INSERT INTO predictions
    (patient_id, prediction, confidence, probabilities, alert, predicted_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        result["prediction"],
        result["confidence"],
        result["risk_score"],
        json.dumps(result["probabilities"]),
        result["alert"],
        json.dumps(result["explanation"]),
        result["predicted_at"],
    ))
    con.commit()
    con.close()

def fetch_latest(patient_id: str) -> dict | None:
    con = sqlite3.connect(DB_PATH)
    row = con.execute("""
    SELECT patient_id, prediction, confidence, probabilities, alert, predicted_at
    FROM predictions
    WHERE patient_id = ?
    ORDER BY id DESC LIMIT 1
    """, (patient_id,)).fetchone()
    con.close()
    if not row:
        return None
    return {
        "patient_id": row[0],
        "prediction": row[1],
        "confidence": row[2],
        "probabilities": json.loads(row[3]),
        "alert": row[4],
        "predicted_at": row[5],
    }

# ============================================================
# FEATURE ENGINEERING  (must match notebook exactly)
# ============================================================

FEATURE_NAMES = [
    # Scores
    "mean_score", "median_score", "score_std", "score_range", "score_slope",
    # Time
    "mean_time", "time_std", "time_range", "time_slope",
    # Trend
    "first_last_diff", "percentage_change", "consistency_score",
    # Combined
    "score_time_ratio", "efficiency_score",
]

def extract_features(scores: list, time: list) -> list:
    sc       = scores
    ti       = time
    t        = list(range(len(sc)))
    slope_sc = float(np.polyfit(t, sc, 1)[0])
    slope_ti = float(np.polyfit(t, ti, 1)[0])
    fld      = float(sc[-1] - sc[0])
    return [
        # Scores
        float(np.mean(sc)),
        float(np.median(sc)),
        float(np.std(sc)),
        float(max(sc) - min(sc)),
        slope_sc,
        # Time
        float(np.mean(ti)),
        float(np.std(ti)),
        float(max(ti) - min(ti)),
        slope_ti,
        # Trend
        fld,
        float((fld / (abs(sc[0]) + 1e-6)) * 100),
        float(1 / (np.std(sc) + 1e-6)),
        # Combined
        float(np.mean(sc) / (np.mean(ti) + 1e-6)),
        float(slope_sc / (slope_ti + 1e-6)),
    ]

# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(proba_dict: dict) -> int:
    dec = proba_dict.get("declining", 0)
    imp = proba_dict.get("improving", 0)
    sta = proba_dict.get("stable",    0)
    raw = (dec * 1.0) - (imp * 0.5) - (sta * 0.3)
    return round(max(0.0, min(1.0, raw)) * 100)

# ============================================================
# ALERT LOGIC
# ============================================================

def build_alert(prediction: str, score_slope: float) -> str:
    if score_slope < -2:
        return "CRITICAL: Rapid cognitive decline — immediate clinical review recommended"
    if prediction == "declining":
        return "WARNING: Cognitive decline detected — schedule follow-up assessment"
    if prediction == "improving":
        return "POSITIVE: Cognitive improvement observed — continue current care plan"
    return "STABLE: No significant cognitive change detected"

# ============================================================
# SCHEMAS
# ============================================================
class Session(BaseModel):
    session_id: int
    score: float
    time_taken: float

class PredictRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    date: str = Field(..., description="Date of assessment (YYYY-MM-DD)")
    sessions: List[Session] = Field(..., description="3 Mind Game sessions")

    @field_validator("sessions")
    @classmethod
    def must_have_three_sessions(cls, v):
        if len(v) != 3:
            raise ValueError(f"Must have exactly 3 sessions, got {len(v)}")
        return v

class Probabilities(BaseModel):
    declining: float
    improving: float
    stable: float

class PredictResponse(BaseModel):
    patient_id: str
    prediction: str
    confidence: float
    probabilities: Probabilities
    alert: str
    predicted_at: str

# ============================================================
# FEATURE ENGINEERING (must match notebook exactly)
# ============================================================
FEATURE_NAMES = [
    "mean_score", "score_slope", "score_std", "score_range",
    "mean_time", "time_slope",
]

def extract_features(scores: list, time: list) -> list:
    t = list(range(len(scores)))
    return [
        float(np.mean(scores)),
        float(np.polyfit(t, scores, 1)[0]), # score_slope — top feature
        float(np.std(scores)),
        float(max(scores) - min(scores)),
        float(np.mean(time)),
        float(np.polyfit(t, time, 1)[0]), # time_slope
    ]

# ============================================================
# ALERT LOGIC
# ============================================================
def build_alert(prediction: str, score_slope: float) -> str:
    if score_slope < -2:
        return "CRITICAL: Rapid cognitive decline — immediate clinical review recommended"
    if prediction == "declining":
        return "WARNING: Cognitive decline detected — schedule follow-up assessment"
    if prediction == "improving":
        return "POSITIVE: Cognitive improvement observed — continue current care plan"
    return "STABLE: No significant cognitive change detected"

# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="NESYAN — Cognitive Decline API",
    description="Predicts cognitive state (declining / improving / stable) from Mind Game session data.",
    version="2.0.0",
)

# ============================================================
# ENDPOINTS
# ============================================================
@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Run prediction for a patient",
    tags=["Prediction"],
)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Accepts the frontend JSON (patient_id, date, sessions),
    runs the model, saves the result to the database, and returns it.
    Expected body:
    ```json
    {
    "patient_id": "1",
    "date": "2026-05-22",
    "sessions": [
    { "session_id": 1, "score": 90, "time_taken": 20 },
    { "session_id": 2, "score": 70, "time_taken": 35 },
    { "session_id": 3, "score": 40, "time_taken": 60 }
    ]
    }
    ```
    """
    # Sort sessions by session_id to guarantee order
    sessions = sorted(req.sessions, key=lambda s: s.session_id)
    scores = [s.score for s in sessions]
    time = [s.time_taken for s in sessions]

    # Extract features & predict
    features = extract_features(scores, time)
    X = pd.DataFrame([features], columns=FEATURE_NAMES)
    prediction = model.predict(X)[0]
    proba_arr = model.predict_proba(X)[0]
    proba_dict = {cls: round(float(p), 4) for cls, p in zip(model.classes_, proba_arr)}
    confidence = round(float(max(proba_arr)), 4)
    score_slope = features[1]
    alert = build_alert(prediction, score_slope)
    predicted_at = datetime.now(timezone.utc).isoformat()

    result = {
        "patient_id": req.patient_id,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": proba_dict,
        "alert": alert,
        "predicted_at": predicted_at,
    }

    save_result(req.patient_id, result)

    return PredictResponse(
        patient_id=result["patient_id"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        probabilities=Probabilities(**result["probabilities"]),
        alert=result["alert"],
        predicted_at=result["predicted_at"],
    )

@app.get(
    "/predict/{patient_id}",
    response_model=PredictResponse,
    summary="Get latest prediction for a patient",
    tags=["Prediction"],
)
def get_latest(patient_id: str) -> PredictResponse:
    row = fetch_latest(patient_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction found for patient_id '{patient_id}'"
        )
    return PredictResponse(
        patient_id=row["patient_id"],
        prediction=row["prediction"],
        confidence=row["confidence"],
        probabilities=Probabilities(**row["probabilities"]),
        alert=row["alert"],
        predicted_at=row["predicted_at"],
    )

@app.get("/health", tags=["System"])
def health():
    return {
        "status":       "ok",
        "model_loaded": True,
        "version":      "3.0.0"   # ✅ Bug #3 Fix — was hardcoded "2.0.0"
    }
