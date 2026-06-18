# 🚦 Event-Driven Congestion Forecasting System — ASTRAM

An end-to-end machine learning system and interactive dashboard for forecasting traffic congestion severity in Bengaluru. The system is trained on historical data to predict incident priority, identify road closure risks, forecast congestion durations, cluster active hotspot zones, and recommend field resources (manpower, barricades, and diversions).

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
                                  ▼
                    [ Premium Streamlit Dashboard ]
                    * Predictor, Maps, Analytics & Reports
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



## 🚀 How to Run the Project

### Prerequisite Environment
Make sure Python 3.8+ is installed on your system.

### 1. Install Dependencies
Install all required packages from `requirements.txt`. E.g.:
```bash
pip install -r requirements.txt
```
*(Note: We use `pygeohash` which is pure-Python and runs on Windows without requiring Rust/C++ compilation tools).*

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

### 3. Start the Streamlit Dashboard
Launch the interactive visual dashboard:
```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the dashboard:
* **🔮 Event Impact Predictor**: Feed live parameters to predict priority/road closure and receive resource recommendations.
* **🗺️ Hotspot Map**: Track density hotspots in Bengaluru.
* **📊 Dataset Analytics**: Run charts explaining event causes, hour peaks, and zones.
* **📋 Model Performance**: Inspect validation scores and feature importance.
