# 🚦 Streamlit Interface User Guide — ASTRAM Forecaster

This guide details all the features, interactive controls, and visual components of the **ASTRAM Event Congestion Forecaster** dashboard, explaining how they work under the hood.

---

## 🎨 Interface Theme & Layout Design

The dashboard uses a custom, responsive design that overrides standard HTML elements with high-fidelity styling:
* **Custom Typography**: Injects the Google Font **Outfit** globally for clean readability and a modern aesthetic.
* **Glassmorphic Cards**: Key metrics are displayed in styled HTML containers featuring subtle translucent background gradients, thin borders, backdrop blurring (`backdrop-filter`), and vertical translation hover animations (`transform: translateY`).
* **Categorical Badges**: Employs colored pill badges (Red for high priority/likely closure, Green for low priority/unlikely closure) to help operators assess risks instantly.
* **Layout Grid**: Uses Streamlit columns to display prediction outputs and resource recommendations side-by-side.

---

## 🧭 Page-by-Page Feature Specifications

### 1. 🔮 Event Impact Predictor & Recommendation Engine
The landing page allows traffic management operators to simulate any real-time event and receive resource dispatch guidelines.

#### A. Input Controls (Left to Right Grid)
* **Event Type**: Dropdown selecting `unplanned` or `planned`.
* **Event Cause**: Dropdown selecting from 14 standard causes (e.g., `vehicle_breakdown`, `accident`, `water_logging`, `public_event`, etc.).
* **Vehicle Type Involved**: Dropdown indicating vehicle classification.
* **Bengaluru Zone**: Dropdown matching Bengaluru's 10 geographical sectors.
* **Corridor**: Dropdown specifying one of the major road corridors or `Non-corridor`.
* **Jurisdiction Police Station**: Text input (e.g., `Cubbon Park`) to specify the handling station.
* **Latitude/Longitude**: Numeric inputs constrained to Bengaluru bounds ($[12.75, 13.30]$ and $[77.25, 77.85]$).
* **Hour of Day**: Slider ($0-23$ hours) representing the incident start time.
* **Day of Week**: Selectbox mapping to the day name.
* **Month**: Slider ($1-12$) representing the calendar month.

#### B. Prediction Engine Pipeline
When the **Predict & Recommend** button is clicked:
1. The coordinates are encoded into a **geohash6** string.
2. The app queries the serialized training lookups:
   * It retrieves historical statistics for the specific geohash cell.
   * It fetches the historical frequency for the specific zone at that hour.
   * It retrieves the risk score for the selected corridor.
3. These features, along with OHE temporal categories, are aligned and passed to the trained LightGBM and XGBoost pipelines.
4. **Duration Regression** predicts duration in a log-scale, which is exponentiated back to minutes (`np.expm1`).
5. **Prediction Logging**: Every prediction is automatically logged to `logs/prediction_log.csv` with all input features and model outputs for monitoring and drift detection.

#### C. Output Metrics & Badges
* **Priority Risk**: Displays the probability of the event being marked high-priority. Includes a green/red **LOW** or **HIGH** priority badge.
* **Road Closure Risk**: Displays the probability of requiring a road closure relative to the tuned decision threshold (e.g., `0.64`). Shows a red **Likely** or green **Unlikely** badge.
* **Est. Duration**: Displays the predicted duration in minutes.
* **Context Zone**: Displays the selected zone and the derived geohash code.

#### D. Operational Resource Panel
Outputs rule-based recommendations mapped from the ML probability scores:
* **Manpower Recommendation**:
  * *High Risk (p ≥ 0.75)*: Recommends 8–12 officers + 2 PCR vans.
  * *Medium Risk (0.45 ≤ p < 0.75)*: Recommends 4–6 officers + 1 PCR van.
  * *Low Risk (p < 0.45)*: Recommends 2 officers.
  * *Special Cause Override*: Public events, protests, and VIP movements automatically trigger high-priority protocol alerts.
* **Barricading Level**:
  * *High Closure Risk (p ≥ 0.60)*: Heavy barricading, route advisory boards, and closure signage.
  * *Medium Closure Risk (0.30 ≤ p < 0.60)*: Partial lane-level barricades.
  * *Low Closure Risk (p < 0.30)*: Cones and soft barricades only.
* **Diversion Urgency**:
  * *High Risk & Peak Commute Hour (7-9 AM, 5-8 PM)*: Recommends urgent alternate routes and Waze/Google Maps updates.
  * *Medium Risk or Est. Duration > 90 mins*: Recommends pre-positioning route advisory boards.
  * *Low Risk*: Recommends simple active monitoring.

#### E. Priority Gauge Chart
A radial Plotly gauge displaying the Priority Risk from 0% to 100%. The gauge arc is divided into color-coded bands (Green: 0–40% Low, Orange: 40–70% Medium, Red: 70–100% High) to provide a clear, glanceable warning system.

---

### 2. 🗺️ Hotspot Map
Visualizes dense, historical congestion pockets to guide preventive police positioning.

* **Plotly Mapbox Scatter Plot**:
  * Draws historical incident points categorized into DBSCAN density clusters.
  * Includes zoom, pan, and hover tooltip details displaying the primary cause and priority level.
  * Uses a clean `carto-positron` map style (no tokens required).
* **Cluster Profiles Table**:
  * Displays a table of all active DBSCAN clusters.
  * Columns include: Cluster ID, Centroid Latitude/Longitude, Total Incident Count, percentage of High Priority incidents, and the dominant cause of congestion in that cluster.

---

### 3. 📊 Dataset Analytics
Exposes exploratory data analysis on the raw historical dataset.

* **Overview KPI Cards**: Display totals for all logged events, unplanned incidents, high-priority reports, and road closures.
* **Events by Cause Bar Chart**: Ranks the top primary causes of congestion.
* **Congestion Peak Hour Line Chart**: Visualizes traffic congestion peaks over a 24-hour cycle.
* **Share of Events by Zone Pie Chart**: A donut chart illustrating the percentage distribution of traffic events across Bengaluru.
* **Priority Casing Stacked Bar Chart**: Segments event counts within each cause by low/high priority categories.

---

### 4. 📋 Model Performance Report
Provides model metrics to support transparency in deployment.

* **Model Performance Tabs**:
  * **Priority Model Tab**: Displays validation precision, recall, F1-score, and ROC-AUC (~0.95) for the LightGBM classifier.
  * **Road Closure Model Tab**: Displays validation metrics for the tuned XGBoost model, explaining the threshold selection to handle class imbalance.
  * **Duration Regressor Tab**: Displays Mean Absolute Error (MAE) and R² statistics alongside validation details.
* **Feature Importance Chart**:
  * Displays a horizontal bar chart ranking the top 10 most influential features used by the LightGBM model.

---

### 5. 💬 ASTRAM Assistant (Chatbot)
A conversational AI chatbot that allows traffic control operators to interact with the system using natural language instead of navigating forms.

#### A. Chat Interface
* Built using Streamlit's native `st.chat_input()` and `st.chat_message()` components for a modern messaging-style experience.
* **Persistent chat history**: Messages are preserved in `st.session_state` across interactions within the same session.
* **Clear Chat History button**: Available in the sidebar to reset the conversation.
* Custom CSS styling applies rounded message bubbles for a polished chat appearance.

#### B. Intent Classification Engine
The chatbot uses a **rule-based NLU pipeline** (no external API keys needed) implemented in `chatbot.py`. When a user sends a message, it is processed through this pipeline:

1. **Greeting/Help Detection**: Matches words like `"help"`, `"hi"`, `"hello"`, `"?"` → returns capability list.
2. **Status/Metrics Detection**: Matches words like `"accuracy"`, `"performance"`, `"f1"`, `"auc"`, `"mae"` → returns model evaluation metrics.
3. **Compare Detection**: Matches `"compare"` or `"vs"` → extracts corridor and/or zone names → returns risk comparison table.
4. **Hotspot Detection**: Matches `"hotspot"`, `"cluster"`, `"dangerous area"` → optionally extracts zone filter → returns top 5 DBSCAN clusters.
5. **Explain Detection**: Matches `"why"`, `"explain"`, `"how does"` → returns topic-specific explanation (priority, closure, duration, or recommendations).
6. **Predict Detection**: Extracts lat/lon (regex), hour (AM/PM parsing), cause (synonym dictionary with 30+ entries), zone, corridor, and day → runs all 3 ML models → returns formatted prediction + recommendation table.
7. **Fallback**: If no intent matches, returns the help message with example queries.

#### C. Smart Defaults
When the chatbot detects a predict intent but the user hasn't specified all parameters, it:
* Auto-fills missing values with sensible defaults (current hour, Majestic coordinates, Central Zone 2, etc.)
* Appends an informational note telling the user which defaults were applied
* Example: `"accident at 5pm"` → uses default location, zone, and corridor but honors the specified hour and cause

#### D. Response Formatting
All responses use **Markdown tables** for clean, structured output:
* Predictions show priority risk, closure risk, duration, location, and time context
* Recommendations show manpower, barricading, and diversion in a separate table
* Hotspot queries return top clusters with event count, priority rate, centroid, and cause
* Corridor comparisons show high-priority rate and risk level (🔴 HIGH / 🟡 MEDIUM / 🟢 LOW)

#### E. Example Interactions

| User Says | Chatbot Response |
|---|---|
| `"accident at 12.97, 77.59 at 5pm"` | Runs ML models → Priority: 45.2% LOW, Closure: 12.3% UNLIKELY, Duration: 62 min → recommends 2 officers, cones, monitor |
| `"waterlogging near Central Zone 2 at 8am on Monday"` | Runs predictions with extracted cause, zone, hour, and day |
| `"show hotspots"` | Returns top 5 DBSCAN clusters sorted by event count |
| `"hotspots in North Zone 1"` | Filters clusters to North Zone 1 only |
| `"compare Mysore Road vs Hosur Road"` | Returns corridor risk scores side-by-side |
| `"why is priority high for protests?"` | Explains the special-cause override rule |
| `"how are recommendations generated?"` | Returns the full recommendation rule table |
| `"model accuracy"` | Returns ROC-AUC, F1, MAE, R² for all 3 models |
| `"hello"` | Returns capability list with example queries |

---

### 6. 📡 Model Monitoring & Drift Detection
Provides real-time visibility into model health, prediction quality, and online learning status.

#### A. Prediction Volume Overview
Four KPI metric cards displaying:
* **Total Predictions**: Cumulative count of all predictions logged since deployment.
* **Last 24 Hours**: Number of predictions made in the most recent 24-hour window.
* **Avg Priority Risk**: Mean priority probability across all predictions.
* **Avg Closure Risk**: Mean closure probability across all predictions.

#### B. Prediction Distribution Histograms
Two side-by-side Plotly histograms (red and blue themed) showing:
* **Priority Risk Distribution**: Histogram of all predicted priority probabilities (0–100%), useful for detecting if the model is suddenly predicting higher/lower risk than normal.
* **Closure Risk Distribution**: Histogram of all predicted closure probabilities.

These histograms serve as a visual early warning for **prediction drift** — if the distribution shape changes significantly, it may indicate the model is encountering data patterns it wasn't trained on.

#### C. PSI Drift Detection Table
When at least 40 predictions have been logged, the system runs **Population Stability Index (PSI)** drift detection:
* Splits the prediction log into two halves (early vs. recent)
* Computes PSI for priority probability, closure probability, and estimated duration
* Displays a table with PSI values and traffic-light status:
  * ✅ **STABLE** (PSI < 0.10) — No action needed
  * ⚠️ **DRIFTING** (PSI 0.10–0.25) — Monitor closely; consider investigating
  * 🔴 **SIGNIFICANT_SHIFT** (PSI > 0.25) — Retraining recommended

If fewer than 40 predictions exist, a warning message is shown indicating the minimum threshold.

#### D. Recent Prediction Log
A scrollable dataframe showing the **last 20 predictions** with:
* Timestamp
* Event cause
* Zone
* Priority probability
* Closure probability
* Estimated duration (minutes)

This provides audit trail visibility and helps operators verify that the system is responding correctly.

#### E. Online Learning Status Panel
Displays the most recent incremental retraining result from `models/retrain_log.json`:
* **Last Retrain**: Timestamp of the most recent retrain attempt
* **Status**: `STARTED`, `READY`, `SKIPPED`, or `ERROR`
* **Data Rows**: Number of data rows used in the retrain window

If no retraining has been performed, displays instructions for running the incremental retrain script:
```bash
python online_update.py --retrain --window-months 6
```
