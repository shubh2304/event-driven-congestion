# 🚦 ASTRAM Event-Driven Congestion Forecasting System — Features Guide

Welcome to the feature documentation for the **ASTRAM Event-Driven Congestion Forecasting System**. This document details all major capabilities, algorithms, and models powering this system.

---

## 🗺️ System Architecture Overview

The system is designed as a modular predictive pipeline that maps raw, continuous traffic events into actionable, real-time dispatch and management recommendations. It consists of a backend data processing and modeling suite, a local NLU conversational engine, drift monitoring, online retrainers, a FastAPI gateway, and two interactive interfaces (a Next.js React application and a Streamlit dashboard).

```
                      +---------------------------------------+
                      |       Raw ASTRAM Traffic Events       |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |      Data Preprocessing Pipeline      |
                      |   (Cleaning, Imputation, Geospatial)  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |      Multi-Model Prediction Layer     |
                      | * Priority Risk Classifier (LightGBM) |
                      | * Road Closure Risk (XGBoost)         |
                      | * Duration Estimator (LGBM Regressor) |
                      +-------------------+-------------------+
                                          |
                                          v
                      +-------------------+-------------------+
                      |          Recommendation Engine        |
                      |      (Manpower, Barricades, Route)    |
                      +-------------------+-------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                 |                                 |
        v                                 v                                 v
+-------+-------+                 +-------+-------+                 +-------+-------+
|  FastAPI REST |                 | Streamlit App |                 |  Next.js App  |
|  (Port 8000)  |                 |  (Port 8501)  |                 |  (Port 3000)  |
+-------+-------+                 +---------------+                 +---------------+
        |
        v
+-------+-------------------------------------------------------------------+
|  Production Utilities: Logging, Drift Detection, Online Retraining       |
+---------------------------------------------------------------------------+
```

---

## 🔮 1. Multi-Model Predictive Engine

At the core of the forecasting pipeline are three machine learning models, trained on historical Bengaluru incident data to output independent congestion risk metrics.

### A. Priority Risk Classifier (LightGBM)
* **Objective**: Predicts the probability of an incident being classified as **High Priority** (e.g. major traffic hazards, protests, multi-vehicle crashes).
* **Model Type**: LightGBM Classifier.
* **Class Imbalance Mitigation**: Implements Synthetic Minority Over-sampling Technique (`SMOTE`) during training to handle skewed class distributions.
* **Accuracy**: Near-perfect classification alignment (Weighted F1 ~99.9%) on standard test splits.

### B. Road Closure Risk Classifier (XGBoost)
* **Objective**: Predicts whether an incident will require closing the road or redirecting traffic.
* **Model Type**: XGBoost Classifier.
* **Imbalance Handling**: Road closures represent only 8.2% of the dataset. The training pipeline uses `scale_pos_weight` and tunes the prediction threshold to **0.638** to maximize minority class F1. This maximizes recall (52%) while preserving high overall accuracy (91%).

### C. Estimated Duration Regressor (LightGBM)
* **Objective**: Estimates the total duration of the congestion event in minutes.
* **Model Type**: LightGBM Regressor.
* **Target Scaling**: The duration target `duration_minutes` is log-transformed (`log1p`) during training to stabilize highly skewed, heavy-tailed duration values, then exponentiated (`np.expm1`) at inference.
* **Performance**: Yields a Mean Absolute Error (MAE) of ~53 minutes.

---

## 📋 2. Rule-Based Operational Recommendation Engine

To translate raw probabilities into field actions, the system feeds predictions into a rule-based advisory engine.

| Decision Target | Metric Trigger | Dispatch Recommendation |
|---|---|---|
| **Manpower Level** | $p(\text{Priority}) \ge 0.75$ | **HIGH**: Deploy 8–12 officers + 2 PCR vans |
| | $0.45 \le p < 0.75$ | **MEDIUM**: Deploy 4–6 officers + 1 PCR van |
| | $p < 0.45$ | **LOW**: Deploy 2 officers |
| **Barricading Adequacy** | $p(\text{Closure}) \ge 0.60$ | **FULL ROAD CLOSURE**: Heavy barricading, route signs, closure boards |
| | $0.30 \le p < 0.60$ | **PARTIAL**: Lane-level barricades |
| | $p < 0.30$ | **MINIMAL**: Cones / soft barricades only |
| **Diversion Urgency** | $p(\text{Closure}) \ge 0.60$ & Peak Hours | **URGENT**: Reroute traffic via alternative lanes. Update GPS services (Google Maps, Waze). |
| | $p(\text{Closure}) \ge 0.45$ OR Duration > 90m | **RECOMMENDED**: Pre-position diversion/route advisory boards |
| | Otherwise | **MONITOR**: Simple active monitoring |

### 🚨 Special Override Causes
Certain incident categories automatically bypass model predictions due to safety-critical protocols:
* **Causes**: `public_event`, `procession`, `protest`, `vip_movement`.
* **Override Action**: Automatically triggers a **HIGH** manpower alert and recommends **FULL ROAD CLOSURE** for public safety coordination.

---

## 🤖 3. Rule-Based NLU Conversational Chatbot (ASTRAM Assistant)

The assistant (`chatbot.py`) parses natural language queries from operators and translates them into model predictions or database lookups, without requiring external paid LLM APIs.

```
                         Operator Text Input
                                  |
                                  v
                  +---------------+---------------+
                  |       Intent Classifier       |
                  |     (Keywords & Regex)        |
                  +---------------+---------------+
                                  |
            +---------------------+---------------------+
            |                     |                     |
            v                     v                     v
   [ Intent: predict ]   [ Intent: hotspot ]   [ Intent: compare ]
   * Extract coords      * Parse Zone filter   * Parse corridors
   * Match Cause synonyms* Retrieve DBSCAN     * Compare risk rates
   * Auto-fill defaults    clusters            * Render Markdown
   * Run ML Models                               table
```

### Key Intents
1. **`predict`**: *"accident at 12.9716, 77.5946 at 5pm on Monday"*
   * Runs the lat/lon regex, maps "accident" to the canonical cause, extracts the hour, and runs all 3 ML models to return predictions and dispatch recommendations.
2. **`hotspot`**: *"show hotspots in Central Zone 2"*
   * Filters and displays DBSCAN spatial clusters sorted by density.
3. **`compare`**: *"compare Mysore Road vs Hosur Road"*
   * Compares the risk metrics of two key corridors side-by-side.
4. **`status`**: *"show model accuracy"*
   * Renders the current validation performance from `eval_metrics.json`.
5. **`help`**: Returns the list of commands.

### Smart Parameter Auto-Filling
If fields are omitted in a prediction query, the chatbot applies default parameters:
* **Location**: Defaults to Majestic (12.9716, 77.5946).
* **Hour**: Defaults to the current system hour.
* **Cause**: Defaults to `congestion`.
* **Zone**: Defaults to `Central Zone 2`.
* **Corridor**: Defaults to `Non-corridor`.

---

## 🗺️ 4. DBSCAN Congestion Hotspot Map

Identifies physical density clusters where traffic incidents frequently concentrate, assisting with preventive resource planning.

* **Algorithm**: DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
* **Distance Metric**: `haversine` (spherical radians) to measure physical distance accurately over the Earth's curvature.
* **Neighborhood Parameter (`eps`)**: Corresponds to $\approx 550$ meters.
* **Min Samples**: 10.
* **Cluster Profiling**: Computes centroid coordinates, incident counts, percentage of high-priority events, and identifies the dominant incident cause within each cluster.

---

## 📡 5. Population Stability Index (PSI) Drift Detection

Drift detection checks whether production data has shifted from the historical distribution the models were trained on.

$$PSI = \sum \left( P_{\text{current}} - P_{\text{reference}} \right) \times \ln\left( \frac{P_{\text{current}}}{P_{\text{reference}}} \right)$$

* **Reference Set**: The initial half of prediction logs.
* **Current Set**: The recent half of prediction logs.
* **PSI Interpretation**:
  * **$\text{PSI} < 0.10$**: ✅ **STABLE** — Prediction distributions are consistent.
  * **$0.10 \le \text{PSI} \le 0.25$**: ⚠️ **DRIFTING** — Distribution shows mild shift. Monitor closely.
  * **$\text{PSI} > 0.25$**: 🔴 **SIGNIFICANT SHIFT** — Action recommended. Consider retraining the model.

---

## 🔄 6. Sliding-Window Online Retraining

The system provides a framework (`online_update.py`) to keep model boosters fresh as Bengaluru's traffic infrastructure evolves.

1. **Sliding Time Window**: Retrains models on the last $N$ months of event data (default: 6).
2. **Warm-Start Continuation**: Uses LightGBM's continuation training (`init_model` parameter) to update model parameters incrementally without full, resource-heavy training.
3. **Validation Gate**: The updated model is evaluated on a test split. It is only written to disk if performance metrics remain within $5\%$ tolerance of the current production scores. If the new model degrades beyond that, it is rejected, and a warning is logged.

---

## ⚡ 7. Incremental Lookup Table Updates (EMA Blending)

At training time, historical statistics are compiled into lookups (`geohash_lookup.pkl`, `zone_hour_lookup.pkl`, `corridor_risk_lookup.pkl`). When new events are ingested, lookups are updated in real time using an **Exponential Moving Average (EMA)** blending factor:

$$\text{Value}_{\text{new}} = \alpha \times \text{Value}_{\text{incoming}} + (1 - \alpha) \times \text{Value}_{\text{existing}}$$

* **Blending Factor ($\alpha$)**: $0.3$ (weights new data at 30%, old at 70%).
* **New Locations**: Unseen geohashes are registered instantly with their first metrics, preventing cold-start issues during live predictions.

---

## 🖥️ 8. User Interfaces & Integration

### A. Next.js Web App (`http://localhost:3000`)
A modern, glassmorphic visual interface built with React and Next.js:
* **Real-time Prediction Form**: Features selectors for cause, vehicle type, zone, corridor, latitude, longitude, day, and hour.
* **Interactive Gauge Charts**: Renders visual gauges for risk levels.
* **Operational Cards**: Displays clear, color-coded badges indicating dispatch manpower, barricades, and detour routes.
* **Analytics Page**: Integrates charts illustrating traffic volume spikes and distributions.

### B. Streamlit Dashboard (`http://localhost:8501`)
An alternative pythonic dashboard featuring:
* **Interactive Maps**: Renders DBSCAN cluster overlays on a zoomable Mapbox layer.
* **Analytics Tabs**: Displays charts for exploratory data analysis (EDA).
* **Model Performance tab**: Exposes precision, recall, F1, and SHAP-based feature importance.
* **Assistant Panel**: Built-in chat panel for natural language queries.
* **Drift Monitoring**: Interactive graphs of prediction logs, volume indicators, and real-time PSI drift scores.
