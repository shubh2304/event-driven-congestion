"""
ASTRAM REST API — FastAPI prediction and ingestion endpoints.

Decouples ML predictions from the Streamlit UI so external systems
(mobile apps, IoT sensors, CCTV, Waze/Google Maps) can consume them.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict     — Run ML models and return predictions + recommendations
    POST /ingest      — Buffer a new labeled event for incremental retraining
    GET  /health      — Health check with model status
    GET  /metrics     — Return current model evaluation metrics
    GET  /drift       — Run drift detection on prediction logs
"""

import os
import numpy as np
import pandas as pd
import joblib
import json
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from online_update import PredictionLogger, DriftDetector

# ─── GEOHASH ENCODER (same fallback chain as app.py) ───────────
try:
    import geohash2 as gh
    def encode_geohash(lat, lon, precision=6):
        return gh.encode(lat, lon, precision)
except ImportError:
    try:
        import pygeohash as pgh
        def encode_geohash(lat, lon, precision=6):
            return pgh.encode(lat, lon, precision)
    except ImportError:
        def encode_geohash(latitude, longitude, precision=6):
            lat_interval = (-90.0, 90.0)
            lon_interval = (-180.0, 180.0)
            base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
            geohash = []
            bits = [16, 8, 4, 2, 1]
            bit = 0
            ch = 0
            even = True
            while len(geohash) < precision:
                if even:
                    mid = (lon_interval[0] + lon_interval[1]) / 2
                    if longitude > mid:
                        ch |= bits[bit]
                        lon_interval = (mid, lon_interval[1])
                    else:
                        lon_interval = (lon_interval[0], mid)
                else:
                    mid = (lat_interval[0] + lat_interval[1]) / 2
                    if latitude > mid:
                        ch |= bits[bit]
                        lat_interval = (mid, lat_interval[1])
                    else:
                        lat_interval = (lat_interval[0], mid)
                even = not even
                if bit < 4:
                    bit += 1
                else:
                    geohash.append(base32[ch])
                    bit = 0
                    ch = 0
            return "".join(geohash)

# ─── GLOBALS ────────────────────────────────────────────────────
models = {}
prediction_logger = PredictionLogger()


# ─── LIFESPAN (model loading) ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, release at shutdown."""
    global models
    try:
        models = {
            'priority':          joblib.load('models/priority_classifier.pkl'),
            'closure':           joblib.load('models/closure_classifier.pkl'),
            'duration':          joblib.load('models/duration_regressor.pkl'),
            'cluster_profile':   joblib.load('models/cluster_profiles.pkl'),
            'cluster_points':    joblib.load('models/cluster_points.pkl'),
            'features_priority': joblib.load('models/feature_list_priority.pkl'),
            'features_closure':  joblib.load('models/feature_list_closure.pkl'),
            'features':          joblib.load('models/feature_list.pkl'),
            'geohash_lookup':    joblib.load('models/geohash_lookup.pkl'),
            'zone_hour_lookup':  joblib.load('models/zone_hour_lookup.pkl'),
            'corridor_risk_lookup': joblib.load('models/corridor_risk_lookup.pkl'),
            'cause_closure_lookup': joblib.load('models/cause_closure_lookup.pkl'),
            'cause_duration_lookup': joblib.load('models/cause_duration_lookup.pkl'),
            'global_medians':    joblib.load('models/global_medians.pkl'),
            'closure_best_thresh': joblib.load('models/closure_best_threshold.pkl'),
            'preprocessor':      joblib.load('models/preprocessor.pkl'),
            'eval_metrics':      json.load(open('models/eval_metrics.json')),
        }
        print("[OK] All models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        print("   Run `python train_pipeline.py` first.")
    yield
    models.clear()


# ─── APP SETUP ──────────────────────────────────────────────────
app = FastAPI(
    title="ASTRAM Congestion Forecasting API",
    description="REST API for Bengaluru traffic congestion prediction and resource recommendation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REQUEST / RESPONSE MODELS ─────────────────────────────────
class EventInput(BaseModel):
    latitude: float = Field(12.9716, ge=12.75, le=13.30,
                            description="Latitude within Bengaluru bounds")
    longitude: float = Field(77.5946, ge=77.25, le=77.85,
                             description="Longitude within Bengaluru bounds")
    hour: int = Field(8, ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(0, ge=0, le=6,
                             description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(6, ge=1, le=12, description="Month of year")
    event_cause: str = Field("congestion", description="Event cause category")
    event_type: str = Field("unplanned", description="planned or unplanned")
    veh_type: str = Field("others", description="Vehicle type involved")
    zone: str = Field("Central Zone 2", description="Bengaluru zone")
    corridor: str = Field("Non-corridor", description="Corridor name")
    police_station: str = Field("Cubbon Park", description="Jurisdiction station")


class PredictionResponse(BaseModel):
    priority_risk: float
    priority_label: str
    closure_risk: float
    closure_label: str
    estimated_duration_min: int
    geohash: str
    recommendations: dict
    timestamp: str


class IngestEvent(BaseModel):
    latitude: float
    longitude: float
    hour: int
    event_cause: str
    zone: str
    corridor: str
    priority: str = Field(..., description="'High' or 'Low'")
    requires_road_closure: int = Field(..., ge=0, le=1)
    duration_minutes: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    model_count: int
    timestamp: str


# ─── RECOMMENDATION ENGINE (same as app.py) ────────────────────
def generate_recommendations(priority_prob, closure_prob, duration_est,
                              event_cause, hour, zone, closure_thresh=0.396):
    if priority_prob >= 0.75:
        manpower = "HIGH — Deploy 8-12 officers + 2 PCR vans"
    elif priority_prob >= 0.45:
        manpower = "MEDIUM — Deploy 4-6 officers + 1 PCR van"
    else:
        manpower = "LOW — 2 officers sufficient"

    if closure_prob >= closure_thresh:
        barricading = "FULL ROAD CLOSURE — Deploy heavy barricades, signage, and rerouting boards"
    elif closure_prob >= closure_thresh * 0.5:
        barricading = "PARTIAL — Lane-level barricading recommended"
    else:
        barricading = "MINIMAL — Cones / soft barricades only"

    is_peak = hour in [7, 8, 9, 17, 18, 19, 20]
    if closure_prob >= closure_thresh and is_peak:
        diversion = "URGENT — Activate alternate route NOW. Notify Waze/Google Maps."
    elif closure_prob >= closure_thresh * 0.75 or duration_est > 90:
        diversion = "RECOMMENDED — Pre-position route advisory boards"
    else:
        diversion = "MONITOR — No diversion needed currently"

    if event_cause in ['public_event', 'procession', 'protest', 'vip_movement']:
        manpower = "HIGH — Political/public event protocol. Coordinate with event organizers."
        barricading = "FULL ROAD CLOSURE recommended for public safety"

    return {
        'manpower': manpower,
        'barricading': barricading,
        'diversion': diversion,
        'estimated_duration_min': round(duration_est),
        'priority_score': f"{priority_prob * 100:.1f}%",
        'closure_risk': f"{closure_prob * 100:.1f}%"
    }


# ─── ENDPOINTS ──────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(event: EventInput):
    """Run ML models and return predictions with resource recommendations."""
    if not models:
        raise HTTPException(status_code=503,
                            detail="Models not loaded. Run train_pipeline.py first.")

    # Build feature vector
    hour_bins = [-1, 5, 11, 16, 20, 24]
    hour_labels = ['night', 'morning', 'afternoon', 'evening', 'late_evening']
    tod = pd.cut([event.hour], bins=hour_bins, labels=hour_labels)[0]

    geohash_val = encode_geohash(event.latitude, event.longitude, precision=6)

    # Geo lookups
    gl = models['geohash_lookup']
    gm = models['global_medians']
    if geohash_val in gl:
        g = gl[geohash_val]
        geo_event_count = g.get('geo_event_count', gm['geo_event_count'])
        geo_high_priority_rate = g.get('geo_high_priority_rate', gm['geo_high_priority_rate'])
        geo_closure_rate = g.get('geo_closure_rate', gm['geo_closure_rate'])
        geo_avg_duration = g.get('geo_avg_duration', gm['geo_avg_duration'])
    else:
        geo_event_count = gm['geo_event_count']
        geo_high_priority_rate = gm['geo_high_priority_rate']
        geo_closure_rate = gm['geo_closure_rate']
        geo_avg_duration = gm['geo_avg_duration']

    # Zone-hour lookup
    zh_key = (event.zone, event.hour)
    zhl = models['zone_hour_lookup']
    zone_hour_event_count = (zhl[zh_key].get('zone_hour_event_count', gm['zone_hour_event_count'])
                             if zh_key in zhl else gm['zone_hour_event_count'])

    # Corridor risk
    crl = models['corridor_risk_lookup']
    ck = event.corridor.strip().lower()
    corridor_risk_score = (crl[ck].get('corridor_risk_score', gm['corridor_risk_score'])
                           if ck in crl else gm['corridor_risk_score'])

    # Cause closure rate lookup (RC-1)
    ccl = models['cause_closure_lookup']
    cause_closure_rate = ccl.get(event.event_cause, gm['cause_closure_rate'])

    # Cause average duration lookup (DR-1)
    cdl = models['cause_duration_lookup']
    cause_avg_duration = cdl.get(event.event_cause, gm['cause_avg_duration'])

    input_dict = {
        'is_planned': int(event.event_type == 'planned'),
        'hour': event.hour,
        'day_of_week': event.day_of_week,
        'month': event.month,
        'is_weekend': int(event.day_of_week >= 5),
        'is_peak_hour': int(event.hour in [7, 8, 9, 17, 18, 19, 20]),
        'latitude': event.latitude,
        'longitude': event.longitude,
        'event_cause': event.event_cause,
        'veh_type': event.veh_type,
        'corridor': event.corridor,
        'police_station': event.police_station,
        'zone': event.zone,
        'junction': 'unknown_junction',
        'geo_event_count': geo_event_count,
        'geo_high_priority_rate': geo_high_priority_rate,
        'geo_closure_rate': geo_closure_rate,
        'geo_avg_duration': geo_avg_duration,
        'zone_hour_event_count': zone_hour_event_count,
        'corridor_risk_score': corridor_risk_score,
        'cause_closure_rate': cause_closure_rate,
        'cause_avg_duration': cause_avg_duration,
        'is_high_risk_cause': int(cause_closure_rate >= 0.3),
    }

    for dummy_col in models['preprocessor']['time_dummies']:
        label = dummy_col.replace('time_of_day_', '')
        input_dict[dummy_col] = int(str(tod) == label)

    input_df = pd.DataFrame([input_dict])

    def align(base_df, feat_list):
        df = base_df.copy()
        for c in feat_list:
            if c not in df.columns:
                df[c] = 0
        return df[feat_list]

    df_prio = align(input_df, models['features_priority'])
    df_closure = align(input_df, models['features_closure'])
    df_all = align(input_df, models['features'])

    prio_prob = float(models['priority'].predict_proba(df_prio)[0][1])
    close_prob = float(models['closure'].predict_proba(df_closure)[0][1])
    try:
        dur_log = models['duration'].predict(df_all)[0]
        dur_est = max(1.0, float(np.expm1(dur_log)))
    except Exception as e:
        logging.warning(f"Duration prediction failed: {e}")
        dur_est = 60.0

    thresh = models['closure_best_thresh']
    recs = generate_recommendations(prio_prob, close_prob, dur_est,
                                     event.event_cause, event.hour,
                                     event.zone, closure_thresh=thresh)

    # Log prediction
    prediction_logger.log_prediction(
        input_params=input_dict,
        predictions={'priority_prob': prio_prob, 'closure_prob': close_prob,
                     'duration_est': dur_est},
        recommendations=recs
    )

    return PredictionResponse(
        priority_risk=round(prio_prob, 4),
        priority_label="HIGH" if prio_prob > 0.5 else "LOW",
        closure_risk=round(close_prob, 4),
        closure_label="LIKELY" if close_prob >= thresh else "UNLIKELY",
        estimated_duration_min=round(dur_est),
        geohash=geohash_val,
        recommendations=recs,
        timestamp=datetime.now().isoformat()
    )


@app.post("/ingest")
async def ingest_event(event: IngestEvent):
    """Buffer a new labeled event for incremental lookup updates and retraining."""
    event_df = pd.DataFrame([event.model_dump()])

    # Update geohash lookup incrementally
    from online_update import LookupUpdater
    updater = LookupUpdater()
    updated = updater.update_geohash_lookup(event_df, encode_geohash)

    return {
        "status": "ingested",
        "geohash_cells_updated": updated,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if models else "degraded",
        models_loaded=bool(models),
        model_count=len(models),
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics")
async def get_metrics():
    """Return current model evaluation metrics."""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    return models.get('eval_metrics', {})


@app.get("/drift")
async def check_drift():
    """Run drift detection on prediction logs."""
    logs = prediction_logger.get_all_predictions()
    if logs.empty:
        return {"status": "no_data", "message": "No prediction logs yet."}

    detector = DriftDetector()
    pred_drift = detector.check_prediction_drift(logs)
    return {
        "prediction_drift": pred_drift,
        "total_predictions": len(logs),
        "timestamp": datetime.now().isoformat()
    }


# ─── NEW ENDPOINTS FOR NEXT.JS FRONTEND ────────────────────────

from chatbot import parse_user_message, execute_intent


class ChatInput(BaseModel):
    message: str = Field(..., description="User message text")


@app.post("/chatbot")
async def chatbot(chat_input: ChatInput):
    """Process a chatbot message and return the response."""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    parsed = parse_user_message(chat_input.message)
    response = execute_intent(
        parsed, models,
        encode_geohash_fn=encode_geohash,
        generate_recommendations_fn=generate_recommendations
    )
    return {
        "response": response,
        "intent": parsed.intent,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/hotspots")
async def get_hotspots():
    """Return cluster profiles and cluster points for the hotspot map."""
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    cp = models.get('cluster_profile')
    pts = models.get('cluster_points')

    if cp is None or pts is None:
        raise HTTPException(status_code=404, detail="Cluster data not available.")

    # Filter to clustered points only (cluster >= 0)
    clustered = pts[pts['cluster'] >= 0].copy()

    # Replace NaN values with None to prevent serialization errors
    cp = cp.replace({np.nan: None})
    clustered = clustered.replace({np.nan: None})

    # Convert to JSON-safe format
    profiles = cp.to_dict(orient='records')
    points = clustered[['latitude', 'longitude', 'cluster', 'event_cause', 'priority']].to_dict(orient='records')

    return {
        "profiles": profiles,
        "points": points,
        "total_clusters": len(cp),
        "total_points": len(points)
    }


# Cache for analytics data
_analytics_cache = {}


@app.get("/analytics")
async def get_analytics():
    """Return dataset analytics computed from the raw CSV."""
    global _analytics_cache
    if _analytics_cache:
        return _analytics_cache

    # Load raw data
    csv_path = "Astram_event_data_anonymized.csv"
    if not os.path.exists(csv_path):
        candidates = [f for f in os.listdir('.') if 'Astram event data_anonymized' in f and f.endswith('.csv')]
        if candidates:
            csv_path = candidates[0]
        else:
            raise HTTPException(status_code=404, detail="Dataset CSV not found.")

    df = pd.read_csv(csv_path)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], utc=True, errors='coerce')
    df['hour'] = df['start_datetime'].dt.hour

    # Overview stats
    total_events = len(df)
    unplanned = int((df['event_type'] == 'unplanned').sum())
    high_priority = int((df['priority'] == 'High').sum())
    road_closures = int(df['requires_road_closure'].sum())

    # Cause distribution
    cause_counts = df['event_cause'].str.strip().str.lower().value_counts().reset_index()
    cause_counts.columns = ['cause', 'count']
    cause_data = cause_counts.to_dict(orient='records')

    # Hourly distribution
    hourly = df.groupby('hour').size().reset_index(name='count')
    hourly_data = hourly.to_dict(orient='records')

    # Zone distribution
    zone_data_raw = df.groupby('zone').size().reset_index(name='count').dropna()
    zone_data = zone_data_raw.to_dict(orient='records')

    # Priority by cause
    cause_priority = df.groupby(['event_cause', 'priority']).size().reset_index(name='count')
    priority_by_cause = cause_priority.to_dict(orient='records')

    _analytics_cache = {
        "overview": {
            "total_events": total_events,
            "unplanned": unplanned,
            "high_priority": high_priority,
            "road_closures": road_closures
        },
        "cause_distribution": cause_data,
        "hourly_distribution": hourly_data,
        "zone_distribution": zone_data,
        "priority_by_cause": priority_by_cause
    }

    return _analytics_cache


@app.get("/monitoring/predictions")
async def get_monitoring_predictions():
    """Return prediction logs with volume statistics for monitoring."""
    logs = prediction_logger.get_all_predictions()

    if logs.empty:
        return {
            "has_data": False,
            "total": 0,
            "last_24h": 0,
            "avg_priority_risk": 0,
            "avg_closure_risk": 0,
            "recent": [],
            "priority_values": [],
            "closure_values": []
        }

    # Recent predictions (last 24h)
    recent_logs = prediction_logger.get_recent_predictions(hours=24)

    # Average risks
    prio_vals = pd.to_numeric(logs['priority_prob'], errors='coerce').dropna()
    close_vals = pd.to_numeric(logs['closure_prob'], errors='coerce').dropna()

    # Recent log entries for table display
    display_cols = ['timestamp', 'event_cause', 'zone', 'priority_prob',
                    'closure_prob', 'duration_est_min']
    available_cols = [c for c in display_cols if c in logs.columns]
    recent_entries = logs[available_cols].tail(20).sort_values(
        'timestamp', ascending=False).to_dict(orient='records')

    return {
        "has_data": True,
        "total": len(logs),
        "last_24h": len(recent_logs),
        "avg_priority_risk": round(float(prio_vals.mean() * 100), 1) if len(prio_vals) > 0 else 0,
        "avg_closure_risk": round(float(close_vals.mean() * 100), 1) if len(close_vals) > 0 else 0,
        "recent": recent_entries,
        "priority_values": [round(float(v * 100), 1) for v in prio_vals.tolist()],
        "closure_values": [round(float(v * 100), 1) for v in close_vals.tolist()]
    }


@app.get("/monitoring/retrain-status")
async def get_retrain_status():
    """Return the online learning retrain status."""
    retrain_log_path = 'models/retrain_log.json'
    if not os.path.exists(retrain_log_path):
        return {"has_retrained": False}

    with open(retrain_log_path, 'r') as f:
        retrain_logs = json.load(f)

    if not retrain_logs:
        return {"has_retrained": False}

    last = retrain_logs[-1]
    return {
        "has_retrained": True,
        "last_retrain": last.get('timestamp', 'N/A')[:19],
        "status": last.get('status', 'N/A'),
        "data_rows": last.get('data_rows', 'N/A')
    }
