# NESYAN — Cognitive Decline AI Model Service
## Backend Integration & API Documentation

Welcome to the **NESYAN** Cognitive Decline Service integration guide. This document is designed to help the backend developer seamlessly integrate, deploy, and communicate with the cognitive decline prediction microservice.

---

## 1. Project Overview & Architecture

**NESYAN** is an AI-powered service that monitors a patient's cognitive state (whether they are **declining**, **improving**, or **stable**) based on their performance across **three consecutive Mind Game sessions**. 

### System Components
*   **FastAPI Engine (`main.py`)**: A high-performance, asynchronous REST API service exposing prediction, health check, and historical prediction retrieval endpoints.
*   **Machine Learning Model (`alzheimers_model.pkl`)**: A pre-trained Scikit-Learn classifier loaded into memory via `joblib`. It uses 6 engineered features extracted from the mind game scores and time taken to predict cognitive trend shifts.
*   **Database (`nesyan.db`)**: A lightweight SQLite database that automatically logs every prediction, confidence level, class probabilities, generated alerts, and timestamps. It operates with zero-configuration and works seamlessly on free tiers (like Render, Railway, or Fly.io).

```mermaid
graph TD
    A[Frontend / Client] -->|POST /predict | B(FastAPI Server)
    A -->|GET /predict/{id}| B
    B -->|Extract 6 Features| C[Feature Pipeline]
    C -->|Run Inference| D[joblib ML Classifier]
    B -->|Save Prediction| E[(SQLite: nesyan.db)]
    B -->|Return Response| A
```

---

## 2. Setting Up the Service Locally

To run this microservice locally, the backend developer will need Python (3.10+ recommended) installed.

### Step 1: Create a Virtual Environment
Navigate to the project root directory and create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
Install all the required machine learning and API libraries specified in `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Dependencies include: `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `pandas`, `numpy`, `joblib`.*

### Step 3: Run the Server
Launch the development server using Uvicorn:
```bash
uvicorn main:app --reload --port 8000
```
* The API will be served at `http://127.0.0.1:8000`
* Automatically generated, interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`
* ReDoc developer-friendly documentation is available at `http://127.0.0.1:8000/redoc`

---

## 3. Environment Configurations

The service is highly flexible and reads configurations from environment variables. In a production environment, configure the following variables:

| Variable | Description | Default Value | Purpose |
| :--- | :--- | :--- | :--- |
| `MODEL_PATH` | Path to the `.pkl` model file | `alzheimers_model.pkl` | Specify where the trained pickle file is located. |
| `DB_PATH` | Path to the SQLite database file | `nesyan.db` | Specify the storage location for persistent sqlite tracking. |

---

## 4. API Endpoint Specifications

The service exposes three distinct REST endpoints. All request bodies and responses are strict JSON payloads validated via Pydantic schemas.

### 4.1. Run Cognitive Prediction
* **Endpoint:** `POST /predict`
* **Content-Type:** `application/json`
* **Description:** Receives cognitive game data from three sessions, runs the machine learning classification model, saves the prediction details to the database, and returns the assessment result.

#### Request JSON Schema
The request must contain exactly three game sessions sorted chronologically by `session_id`.

```json
{
  "patient_id": "string",
  "date": "string (YYYY-MM-DD)",
  "sessions": [
    {
      "session_id": 1,
      "score": 90.0,
      "time_taken": 20.0
    },
    {
      "session_id": 2,
      "score": 70.0,
      "time_taken": 35.5
    },
    {
      "session_id": 3,
      "score": 40.0,
      "time_taken": 60.0
    }
  ]
}
```

*   `patient_id` *(String, Required)*: A unique identifier for the patient.
*   `date` *(String, Required)*: The date of the assessment in `YYYY-MM-DD` format.
*   `sessions` *(Array of Session objects, Required)*: **Must contain exactly 3 session items**. Each session contains:
    *   `session_id` *(Integer)*: Identifier of the session (usually `1`, `2`, and `3`).
    *   `score` *(Float)*: The score achieved by the patient in that session (typically standard range, e.g. `0` to `100`).
    *   `time_taken` *(Float)*: Time spent completing the mind game, in seconds.

#### Response JSON Schema
```json
{
  "patient_id": "string",
  "prediction": "declining | improving | stable",
  "confidence": 0.9421,
  "probabilities": {
    "declining": 0.9421,
    "improving": 0.0034,
    "stable": 0.0545
  },
  "alert": "string",
  "predicted_at": "ISO-8601 Timestamp (UTC)"
}
```

*   `prediction` *(String)*: The predicted cognitive trend class. One of:
    *   `declining`: Patient shows cognitive deterioration over the sessions.
    *   `improving`: Patient shows cognitive enhancement or adaptability.
    *   `stable`: Patient maintains consistent, steady cognitive performance.
*   `confidence` *(Float)*: The probability value of the predicted class (ranges from `0.0` to `1.0`).
*   `probabilities` *(Object)*: The breakdown of probability scores for each of the three target classes.
*   `alert` *(String)*: A priority action message derived from the clinical alert logic (see section below).
*   `predicted_at` *(String)*: The exact UTC time when the prediction was computed, formatted in ISO-8601 standard.

---

### 4.2. Retrieve Latest Assessment
* **Endpoint:** `GET /predict/{patient_id}`
* **Description:** Retrieves the single most recent prediction stored in the database for the given patient.
* **Error Handling:** Returns `404 Not Found` if the patient has no prediction history.

#### Example Request
```http
GET /predict/patient-abc-123 HTTP/1.1
Host: 127.0.0.1:8000
```

#### Example Response (200 OK)
```json
{
  "patient_id": "patient-abc-123",
  "prediction": "declining",
  "confidence": 0.925,
  "probabilities": {
    "declining": 0.925,
    "improving": 0.015,
    "stable": 0.06
  },
  "alert": "WARNING: Cognitive decline detected — schedule follow-up assessment",
  "predicted_at": "2026-05-24T18:00:00.000000+00:00"
}
```

#### Example Response (404 Not Found)
```json
{
  "detail": "No prediction found for patient_id 'patient-abc-123'"
}
```

---

### 4.3. Health Check
* **Endpoint:** `GET /health`
* **Description:** A simple diagnostics endpoint for load balancers or health probes. Verifies that the server is alive and the AI pickle model is successfully loaded in memory.

#### Response (200 OK)
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## 5. Clinical Alert Logic & Feature Engineering

The backend developer should understand how features are computed from raw inputs, as it governs the ML classification and the threshold-based alert generation.

### 5.1. The 6 Engineered Features
When three sessions are posted, `main.py` extracts a 6-feature vector to feed the Scikit-Learn model:
1.  `mean_score`: The arithmetic mean of the three scores.
2.  `score_slope`: The slope of a linear regression line fitted across the session index `[0, 1, 2]` vs the scores `[s1, s2, s3]`. Calculated via NumPy's `polyfit`. This tracks how rapidly their score rises or falls across the sessions.
3.  `score_std`: The standard deviation of the scores (captures stability/variance).
4.  `score_range`: Max score minus min score.
5.  `mean_time`: The arithmetic mean of the time taken (in seconds) across the three sessions.
6.  `time_slope`: The slope of a linear regression line fitted across the session index vs time taken.

### 5.2. Custom Alert Thresholds
Along with the machine learning prediction, the service runs a secondary logic check based on the **`score_slope`** and **`prediction`** to determine severity:

*   **CRITICAL ALERT** (Triggered if `score_slope < -2`):
    *   *Message:* `CRITICAL: Rapid cognitive decline — immediate clinical review recommended`
    *   *Note:* Even if the model outputs stable/improving, a slope decrease steeper than -2 points/session triggers this immediate clinical threshold.
*   **WARNING ALERT** (Triggered if predicted class is `declining`):
    *   *Message:* `WARNING: Cognitive decline detected — schedule follow-up assessment`
*   **POSITIVE ALERT** (Triggered if predicted class is `improving`):
    *   *Message:* `POSITIVE: Cognitive improvement observed — continue current care plan`
*   **STABLE ALERT** (Triggered if predicted class is `stable`):
    *   *Message:* `STABLE: No significant cognitive change detected`

---

## 6. Database Schema Details

The local database (`nesyan.db`) has a single table `predictions`. If it does not exist, FastAPI will automatically initialize it on startup:

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    TEXT    NOT NULL,
    prediction    TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    probabilities TEXT    NOT NULL,   -- Stores the full JSON string of class probabilities
    alert         TEXT    NOT NULL,
    predicted_at  TEXT    NOT NULL
);
```

> [!NOTE]
> The database tracks the historical timeline. If a patient takes the assessment multiple times, multiple rows will be created. The `GET /predict/{patient_id}` query automatically grabs the latest record using `ORDER BY id DESC LIMIT 1`.

---

## 7. Integration Guidelines for Primary Backend

If your primary backend application is written in Node.js (Express, NestJS), Spring Boot, Django, or Laravel, follow these design patterns to integrate with this AI microservice:

1.  **Frontend Orchestration**: 
    When a patient finishes the game, let the main backend server gather the three session results, validate them, and then make an internal HTTP client request to the **NESYAN AI microservice** at `/predict`.
2.  **Handling Patient Identifiers**:
    Send the primary database's user ID or patient record UUID as the `patient_id` when calling `/predict`. This keeps the history linked between the systems.
3.  **CORS & Security**:
    By default, the AI microservice exposes endpoints publicly unless firewall/VPC rules restrict it. It is recommended to run this FastAPI service inside your private network (VPC) so that only your main backend can make requests to it, keeping the patient database secure.
4.  **Connecting to a Remote Database (Optional)**:
    If SQLite is not preferred for multi-instance production deployments (e.g., behind a load balancer where local storage isn't shared), the database logic in `main.py` can easily be modified to connect to a centralized PostgreSQL or MySQL instance. However, for a single-container setup, the SQLite file works robustly.

---

### Integration Test Example (Node.js Axios / Express)

Here is a clean code snippet the backend developer can drop into their primary Node.js server to call the cognitive assessment microservice:

```javascript
const axios = require('axios');

async function evaluatePatientCognitiveState(patientUuid, sessionsArray) {
  const url = process.env.NESYAN_AI_SERVICE_URL || 'http://localhost:8000';
  
  // Format the request according to the FastAPI JSON schema
  const payload = {
    patient_id: patientUuid,
    date: new Date().toISOString().split('T')[0], // YYYY-MM-DD
    sessions: sessionsArray.map((session, index) => ({
      session_id: index + 1,
      score: Number(session.score),
      time_taken: Number(session.timeTaken) // in seconds
    }))
  };

  try {
    const response = await axios.post(`${url}/predict`, payload);
    return response.data; // PredictResponse matching schema
  } catch (error) {
    console.error('Error integrating with NESYAN Cognitive AI service:', error.message);
    throw new Error('Cognitive prediction service is temporarily unavailable.');
  }
}
```
