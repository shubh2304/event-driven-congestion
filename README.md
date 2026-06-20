# 🚦 Event-Driven Congestion Forecasting System — ASTRAM
#SKMKB

An end-to-end machine learning system and interactive dashboard for forecasting traffic congestion severity in Bengaluru. The system is trained on historical data to predict incident priority, identify road closure risks, forecast congestion durations, cluster active hotspot zones, and recommend field resources (manpower, barricades, and diversions). It also features a **conversational AI chatbot**, **online learning infrastructure** for incremental model updates, **drift detection**, and a **REST API** for external system integrations.

---

## 🏗️ Project Architecture & Pipeline

```
                       Raw ASTRAM Dataset CSV
                                  │
                                  ▼
                    [ STEP 1: Data Initial Cleaning ]
                    * Drop 24 columns (>95% missingness)
                    * Normalize event causes & remove tests
                                  │
                                  ▼
                  [ STEP 2: Datetime Feature Engineering ]
                    * Parse UTC timestamps to Datetime
                    * Derive duration_minutes target (Closed - Start)
                    * Extract hour, weekday, month, peak, is_weekend
                                  │
                                  ▼
                  [ STEP 3: Geospatial Feature Engineering ]
                    * Encode lat/lon using geohash6 (~1.2km cell grid)
                    * Calculate geohash-level historic count & rates
                    * Generate zone × hour pressure index
                    * Calculate frequency-weighted corridor risk
                                  │
                                  ▼
                       [ STEP 4: Outlier Capping ]
                    * Filter lat/lon outside Bengaluru bounds
                    * Soft-cap duration outliers using IQR bounds
                                  │
                                  ▼
                   [ STEP 5: Missing Value Imputation ]
                    * Impute categorical columns with Mode / 'unknown'
                    * Impute numeric features with Median
                    * Impute parsed temporal features (hour, month, etc.)
                                  │
                                  ▼
                   [ STEP 6: Target Encoding & Encoding ]
                    * Low-cardinality time_of_day One-Hot encoded
                    * High-cardinality columns Target Encoded during fit
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
  [ Priority (7A) ]       [ Road Closure (7B) ]     [ Duration (7C) ]
  * LightGBM Classifier   * XGBoost Classifier      * LGBM Regressor
  * TargetEncoder + SMOTE * SMOTE + scale_pos_weight* log1p target scale
  * Weighted F1 ~99.9%    * Tuned F1-Threshold      * MAE: ~53 mins
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                    [ Recommendation Rules Layer ]
                    * Rule-based resource allocation
                    * Manpower, barricades & diversion advisory
                                  │
                                  ▼
                      [ Production Metadata Output ]
                    * Save pipeline models & threshold configs
                    * Generate lookups (geohash, zone-hour, corridor)
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
          [ Streamlit ]    [ REST API ]    [ Online Learning ]
          * 6-page dashboard * FastAPI       * Drift Detection
          * Chatbot          * /predict      * Incremental Retrain
          * Monitoring       * /ingest       * Lookup Updates
```

---

## 📂 Project Structure

```
event-driven-congestion/
├── app.py                  # Streamlit dashboard (6 pages)
├── chatbot.py              # Rule-based NLU chatbot engine
├── online_update.py        # Online learning, drift detection, prediction logging
├── api.py                  # FastAPI REST API for external integrations
├── train_pipeline.py       # Full ML training pipeline
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── INTERFACE_GUIDE.md      # Detailed UI/UX specification
├── Astram_event_data_anonymized.csv   # Raw ASTRAM dataset
├── models/                 # Serialized models, lookups, and configs
│   ├── priority_classifier.pkl
│   ├── closure_classifier.pkl
│   ├── duration_regressor.pkl
│   ├── dbscan_clusterer.pkl
│   ├── cluster_profiles.pkl / cluster_points.pkl
│   ├── feature_list.pkl / feature_list_priority.pkl
│   ├── geohash_lookup.pkl / zone_hour_lookup.pkl / corridor_risk_lookup.pkl
│   ├── global_medians.pkl / closure_best_threshold.pkl
│   ├── preprocessor.pkl
│   ├── eval_metrics.json
│   └── retrain_log.json (created by online_update.py)
└── logs/                   # Prediction logs (created at runtime)
    └── prediction_log.csv
```

---

## 📊 Data Preprocessing & Pipeline Steps

### 1. Data Cleaning
* **Junk Column Removal**: Drops 24 columns that are completely empty or have >95% missing values (e.g. `comment`, `resolved_at_address`, `cargo_material`, etc.).
* **Casing & Consistency**: Normalizes free-text inputs in `event_cause` (converting to lower, stripping whitespaces, mapping synonyms) and removes test/demo rows.

### 2. Time-Series Feature Engineering
* Parses timestamps (`start_datetime`, `end_datetime`, `closed_datetime`) to UTC datetimes.
* Derives the **regression target** `duration_minutes` from `closed_datetime` (fallback to `end_datetime`) relative to `start_datetime`.
* Extracts time components: `hour`, `day_of_week` (0-6), `month`, `is_weekend`, `is_peak_hour` (traffic peak slots), and `time_of_day` bins (morning, afternoon, evening, night).

### 3. Geospatial Grid Mapping
* Encodes geographic coordinates into `geohash6` string cells (precision 6 corresponds to cells of $\approx 1.2\text{ km} \times 1.2\text{ km}$).
* Merges location-level historical aggregations:
  * `geo_event_count`: Frequency of incidents in the geohash cell.
  * `geo_high_priority_rate`: Historical ratio of high-priority events in the cell.
  * `geo_closure_rate`: Historical ratio of road closures in the cell.
  * `geo_avg_duration`: Average duration of historical incidents in the cell.
* Computes `zone_hour_event_count` (incident density for a zone at a specific hour).
* Computes `corridor_risk_score` (historical high-priority rates on particular corridors).

### 4. Outlier Removal & Imputation
* Filters coordinate values to fit within Bengaluru's boundary bounds (`latitude` $\in [12.75, 13.30]$, `longitude` $\in [77.25, 77.85]$).
* Applies a **soft cap** to `duration_minutes` outliers using the Interquartile Range (IQR) method, replacing values outside the bounds with the lower/upper thresholds to preserve training size.
* Imputes missing categorical attributes with the mode, and missing numeric aggregations with the median. It also corrects temporal features that fail parsing.

### 5. Multi-Model Pipelines
* **Priority Classifier**: ImbPipeline incorporating `TargetEncoder` + `SMOTE` over-sampling + `LightGBM` (minimizes class imbalance, resulting in a Weighted F1 of **~99.9%**).
* **Road Closure Classifier**: XGBoost pipeline optimized for recall on severe road closure imbalance using `scale_pos_weight` and threshold tuning to maximize minority class F1-score.
* **Duration Regressor**: LGBMRegressor model trained on log-transformed duration target (`log1p(duration_minutes)`) to stabilize skewed duration values.
* **DBSCAN Hotspot Clusterer**: Density-based spatial clustering on spherical radians to identify physical congestion centroids (using an `eps` distance equivalent to ~550 meters).

### 6. Production Lookup System
During training, lookup dictionaries are saved (`geohash_lookup.pkl`, `zone_hour_lookup.pkl`, `corridor_risk_lookup.pkl`) along with the global medians. At inference time, the application takes the latitude/longitude coordinate, calculates the geohash, and retrieves the exact historical context from the lookup tables, falling back to global medians for new/unseen locations.

---

## 📈 Model Evaluation & Operational Efficacy

This section details the validation metrics of our models and highlights how the recommendation engine leverages these scores to suggest field dispatch protocols.

### 1. Model Validation Metrics (After Training)

| Model Target | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC / MAE |
|---|---|---|---|---|---|
| **Priority Classification** (LightGBM) | **100%** | **1.00** | **1.00** | **1.00** | **1.0000** ROC-AUC |
| **Road Closure Classification** (XGBoost) | **91%** | **0.43** | **0.52** | **0.47** | **0.8382** ROC-AUC |
| **Duration Regression** (LGBMRegressor) | — | — | — | — | **53.49 mins** MAE / **0.13** $R^2$ |

* **Priority Forecast Efficacy**: The LightGBM classifier achieves exceptionally high scores (Weighted F1 > 0.999), meaning there is near-zero error in class alignment.
* **Road Closure Imbalance Handling**: Road closures represent only 8.2% of the dataset. Since missing a road closure has a high operational cost (causing severe traffic gridlocks), we tuned the decision threshold to **0.638**, raising recall to **52%** on closures while maintaining **91% overall accuracy**.
* **Duration Regression Constraints**: The duration model has moderate R² (0.13) and a MAE of 53.49 minutes. This is primarily a dataset limitation, as `duration_minutes` can only be derived for ~3,000 events (94% missingness on end time).

### 2. Operational Recommendation Quality

The rule-based Recommendation Engine maps these high-accuracy predictions into concrete field actions. Here is why the recommendations are highly reliable for field deployment:

* **Manpower Matching (Low/Medium/High)**: Recommends police officer count based on predicted priority levels. Special overrides trigger a **HIGH** classification for safety-critical causes like protests, processions, and VIP movements regardless of model output.
* **Barricading Adequacy**: Uses predicted road closure probability to determine heavy vs. light equipment deployment, preventing expensive material over-allocation.
* **Diversion Strategy**: Synthesizes peak-hour traffic indices, estimated duration, and closure probability to determine whether a diversion should be activated. During commute rush hours (7-9 AM, 5-8 PM), even moderate closure probabilities trigger urgent rerouting alerts, helping coordinate with GPS services (like Waze and Google Maps) in advance to minimize area-wide congestion.

---

## 🤖 Chatbot Assistant — NLU Engine

The **ASTRAM Traffic Assistant** (`chatbot.py`) provides a natural-language conversational interface for traffic control operators, removing the need to navigate forms and fill 10+ input fields.

### How It Works

The chatbot uses a **Rule-Based Natural Language Understanding (NLU)** pipeline — no external API keys required:

```
User Input (text)
       │
       ▼
┌─────────────────────────┐
│   Intent Classifier     │
│   (keyword + regex)     │
│                         │
│  1. Match greeting/help │
│  2. Match metrics words │
│  3. Match compare/vs    │
│  4. Match hotspot words │
│  5. Match explain words │
│  6. Extract lat/lon,    │
│     hour, cause, zone   │
│     → predict intent    │
│  7. Fallback → help     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Parameter Extractor   │
│                         │
│  • Lat/Lon: regex       │
│    (\d{2}\.\d+, \d{2})  │
│  • Hour: "5pm" → 17     │
│  • Cause: keyword map   │
│    "accident" → accident│
│  • Zone: fuzzy keyword  │
│  • Corridor: keyword    │
│  • Day: "Monday" → 0    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Response Generator    │
│                         │
│  predict → run ML models│
│  hotspot → filter DBSCAN│
│  compare → lookup scores│
│  explain → return rules │
│  status  → eval_metrics │
│  help    → capability   │
└─────────────────────────┘
```

### Supported Intents

| Intent | Example User Input | What Happens |
|---|---|---|
| `predict` | *"accident at 12.97, 77.59 at 5pm"* | Extracts coordinates, hour, and cause → runs all 3 ML models → returns priority, closure risk, duration, and resource recommendations in a formatted table |
| `hotspot` | *"show hotspots in Central Zone 2"* | Filters DBSCAN cluster profiles → returns top 5 clusters with event counts and priority rates |
| `compare` | *"compare Mysore Road vs Hosur Road"* | Looks up corridor risk scores → returns side-by-side comparison table with risk levels |
| `explain` | *"why is priority high for protests?"* | Returns rule-based explanation of how the recommendation engine works for that specific topic |
| `status` | *"model accuracy?"* | Returns current model evaluation metrics (ROC-AUC, F1, MAE, R²) from `eval_metrics.json` |
| `help` | *"what can you do?"* | Lists all chatbot capabilities with example queries |

### Smart Parameter Handling

The chatbot intelligently **auto-fills defaults** for any parameters not mentioned, and **informs the user** which defaults were applied:
- **Location**: Defaults to Majestic (12.9716, 77.5946) if no lat/lon is provided
- **Hour**: Defaults to the current system hour
- **Event cause**: Defaults to `congestion`
- **Zone**: Defaults to `Central Zone 2`
- **Day/Month**: Defaults to current day and month

### Keyword Dictionaries

The NLU engine includes comprehensive synonym mapping:
* **30+ cause synonyms**: `"crash"` → `accident`, `"flooding"` → `water_logging`, `"bandh"` → `protest`, `"tree fell"` → `tree_fall`, etc.
* **10 zone aliases**: `"central 2"` → `Central Zone 2`, `"north 1"` → `North Zone 1`, etc.
* **10 corridor aliases**: `"mysore"` → `Mysore Road`, `"bellary"` → `Bellary Road 1`, etc.
* **7 day abbreviations**: `"mon"` → Monday (0), `"sat"` → Saturday (5), etc.

---

## 🔄 Online Learning & Drift Detection

The **Online Learning module** (`online_update.py`) provides infrastructure for keeping models fresh as traffic patterns evolve over time.

### Why Online Learning?

| Factor | Justification |
|---|---|
| **Data arrives continuously** | ASTRAM generates new traffic events every day |
| **Distribution shift** | New roads, metro lines, seasonal patterns, and policy changes alter traffic behavior |
| **Retraining cost** | Full retraining on growing datasets becomes expensive over time |
| **Stale lookups** | New geohash cells (new areas of the city) have no historical context |

### Components

#### 1. Prediction Logger (`PredictionLogger`)
Every prediction made through the dashboard or API is logged to `logs/prediction_log.csv`:
```
timestamp, latitude, longitude, hour, day_of_week, month,
event_cause, zone, corridor, priority_prob, closure_prob,
duration_est_min, manpower_level, barricading_level,
diversion_level, actual_priority, actual_closure,
actual_duration_min, feedback_correct
```
* Ground truth fields (`actual_*` and `feedback_correct`) are empty at prediction time and can be filled later when the real outcome is known, enabling supervised retraining.

#### 2. Drift Detector (`DriftDetector`)
Uses **Population Stability Index (PSI)** to detect when model predictions or input features have drifted from their training distributions:

```
PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)
```

| PSI Value | Interpretation |
|---|---|
| < 0.10 | ✅ No significant drift — model is stable |
| 0.10 – 0.25 | ⚠️ Moderate drift — monitor closely |
| > 0.25 | 🔴 Significant drift — consider retraining |

The drift detector compares the first half vs. second half of the prediction log window, measuring PSI for priority probability, closure probability, and estimated duration.

#### 3. Incremental Retrainer (`IncrementalRetrainer`)
Supports **sliding-window micro-retraining**:
1. Loads data from the last N months (configurable, default 6)
2. Runs the same preprocessing pipeline as `train_pipeline.py`
3. Supports LightGBM **continuation training** (warm-start from existing booster weights)
4. **Validation gate**: New models are only promoted if they don't degrade by more than 5% from current metrics — rejected models are logged but not deployed

#### 4. Lookup Updater (`LookupUpdater`)
Incrementally updates geohash lookup tables as new events arrive using **Exponential Moving Average (EMA)** blending:
```
new_value = α × incoming + (1 - α) × existing    (α = 0.3)
```
This keeps lookup tables fresh without requiring a full retrain — new geohash cells are automatically added, and existing cells smoothly incorporate recent data.

### CLI Usage

```bash
# Run incremental retraining on the last 6 months of data
python online_update.py --retrain --window-months 6

# Run drift detection on prediction logs
python online_update.py --drift

# Custom data source and window
python online_update.py --retrain --data path/to/new_data.csv --window-months 3
```

---

## 🌐 REST API (FastAPI)

The **REST API** (`api.py`) decouples ML predictions from the Streamlit UI, enabling external systems (mobile apps, IoT sensors, CCTV systems, Waze/Google Maps) to consume predictions programmatically.

### Starting the API Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Run ML models and return predictions + resource recommendations |
| `/ingest` | POST | Buffer a new labeled event for incremental lookup updates |
| `/health` | GET | Health check — reports model loading status |
| `/metrics` | GET | Returns current model evaluation metrics (`eval_metrics.json`) |
| `/drift` | GET | Runs drift detection on prediction logs and returns PSI values |
| `/docs` | GET | Auto-generated interactive Swagger UI documentation |

### Example: Predict Endpoint

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 12.9716,
    "longitude": 77.5946,
    "hour": 17,
    "day_of_week": 0,
    "month": 6,
    "event_cause": "accident",
    "event_type": "unplanned",
    "veh_type": "private_car",
    "zone": "Central Zone 2",
    "corridor": "Non-corridor",
    "police_station": "Cubbon Park"
  }'
```

**Response:**
```json
{
  "priority_risk": 0.1234,
  "priority_label": "LOW",
  "closure_risk": 0.0567,
  "closure_label": "UNLIKELY",
  "estimated_duration_min": 62,
  "geohash": "tdr1y6",
  "recommendations": {
    "manpower": "LOW — 2 officers sufficient",
    "barricading": "MINIMAL — Cones / soft barricades only",
    "diversion": "MONITOR — No diversion needed currently",
    "estimated_duration_min": 62,
    "priority_score": "12.3%",
    "closure_risk": "5.7%"
  },
  "timestamp": "2026-06-19T12:00:00"
}
```

### Example: Ingest Endpoint (for Online Learning)

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 12.98,
    "longitude": 77.61,
    "hour": 9,
    "event_cause": "accident",
    "zone": "Central Zone 2",
    "corridor": "Hosur Road",
    "priority": "High",
    "requires_road_closure": 1,
    "duration_minutes": 45.0
  }'
```

---

## 📡 Model Monitoring Dashboard

The **Model Monitoring** page (Page 6 in the Streamlit dashboard) provides real-time visibility into model health:

* **Prediction Volume Cards**: Total predictions, last 24-hour count, average priority risk, average closure risk.
* **Distribution Histograms**: Priority and closure risk distributions across all logged predictions — visual check for distribution shift.
* **PSI Drift Detection Table**: Computes Population Stability Index for each prediction output and flags drift status (✅ Stable / ⚠️ Drifting / 🔴 Significant Shift).
* **Recent Prediction Log**: Scrollable table of the last 20 predictions with timestamps, cause, zone, and all prediction outputs.
* **Online Learning Status**: Displays the timestamp, status, and data size of the most recent incremental retrain (from `models/retrain_log.json`).

---

## 🚀 How to Run the Project

### Prerequisite Environment
* Make sure Python 3.8+ is installed on your system.
* Make sure Node.js (v18+) is installed to run the Next.js frontend.

### 1. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```
*(Note: We use `pygeohash` which is pure-Python and runs on Windows without requiring Rust/C++ compilation tools).*

To set up the frontend:
```bash
cd frontend
npm install
```

### 2. Run the Model Training Pipeline
To run the preprocessing steps, train the models, perform cross-validation, and export lookup files:
```bash
python train_pipeline.py
```
This command generates the following artifacts in the `models/` directory:
* `priority_classifier.pkl` / `closure_classifier.pkl` (classification models)
* `duration_regressor.pkl` (regression model)
* `dbscan_clusterer.pkl` / `cluster_profiles.pkl` (spatial clustering files)
* Lookups and configs (`geohash_lookup.pkl`, `zone_hour_lookup.pkl`, `corridor_risk_lookup.pkl`, `global_medians.pkl`, and `closure_best_threshold.pkl`)

### 3. Start the Backend REST API
The Next.js frontend communicates with the FastAPI backend. Start it using:
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```
Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger documentation.

### 4. Start the Next.js Frontend Web App
In a new terminal window, navigate to the `frontend` folder and run:
```bash
cd frontend
npm run dev
```
*(Note: On Windows, if script execution is disabled, you can use `cmd /c npm run dev`).*

Open [http://localhost:3000](http://localhost:3000) in your browser to interact with the Next.js React client application.

### 5. Start the Streamlit Dashboard (Alternative Interface)
If you prefer to run the unified Python Streamlit dashboard instead:
```bash
python -m streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to view pages for Predictor, Hotspot Maps, Analytics, Model Reports, and Chatbot.

### 6. Run Online Learning & Drift Detection
To incrementally retrain models or check for drift:
```bash
# Incremental retrain on the last 6 months
python online_update.py --retrain --window-months 6

# Check drift in prediction logs
python online_update.py --drift
```

---

## 📖 Feature Guide & Explanation
For a detailed guide on the project's models, recommendations, NLU chatbot, drift detection, and lookup architectures, see [FEATURES_EXPLANATION.md](file:///c:/Users/shubh/OneDrive/Desktop/event-driven%20congestion/FEATURES_EXPLANATION.md).

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| ML Models | LightGBM, XGBoost, scikit-learn, imbalanced-learn |
| Feature Engineering | pandas, numpy, pygeohash, category_encoders |
| Dashboard | Streamlit, Plotly |
| Chatbot NLU | Custom rule-based (regex + keyword dictionaries) |
| REST API | FastAPI, Uvicorn, Pydantic |
| Spatial Clustering | DBSCAN (haversine metric) |
| Drift Detection | Population Stability Index (PSI) |
| Online Learning | LightGBM continuation training, EMA lookup updates |
| Model Persistence | joblib, JSON |
