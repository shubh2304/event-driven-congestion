"""
ASTRAM Chatbot — Rule-Based NLU Intent Parser & Response Generator.

Supports natural-language queries from traffic control operators:
    - "accident at 12.97, 77.59 at 5pm"    → runs ML prediction
    - "show hotspots in Central Zone 2"      → filters cluster profiles
    - "compare Mysore Road vs Hosur Road"    → corridor risk comparison
    - "why is priority high for protests?"   → explains recommendation rules
    - "model accuracy?"                      → returns eval metrics summary
    - "help"                                 → lists capabilities
"""

import re
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# ─── CONSTANTS ──────────────────────────────────────────────────
CAUSE_KEYWORDS = {
    'accident': 'accident',
    'crash': 'accident',
    'collision': 'accident',
    'breakdown': 'vehicle_breakdown',
    'vehicle breakdown': 'vehicle_breakdown',
    'broke down': 'vehicle_breakdown',
    'construction': 'construction',
    'road work': 'construction',
    'pothole': 'pot_holes',
    'potholes': 'pot_holes',
    'pot hole': 'pot_holes',
    'waterlogging': 'water_logging',
    'water logging': 'water_logging',
    'flooding': 'water_logging',
    'flood': 'water_logging',
    'public event': 'public_event',
    'event': 'public_event',
    'rally': 'public_event',
    'procession': 'procession',
    'vip': 'vip_movement',
    'vip movement': 'vip_movement',
    'protest': 'protest',
    'dharna': 'protest',
    'bandh': 'protest',
    'tree fall': 'tree_fall',
    'tree fell': 'tree_fall',
    'fallen tree': 'tree_fall',
    'road condition': 'road_conditions',
    'bad road': 'road_conditions',
    'congestion': 'congestion',
    'traffic jam': 'congestion',
    'jam': 'congestion',
    'fog': 'fog_low_visibility',
    'low visibility': 'fog_low_visibility',
    'mist': 'fog_low_visibility',
}

ZONE_KEYWORDS = {
    'central zone 2': 'Central Zone 2', 'central 2': 'Central Zone 2',
    'central zone 1': 'Central Zone 1', 'central 1': 'Central Zone 1',
    'west zone 1': 'West Zone 1', 'west 1': 'West Zone 1',
    'west zone 2': 'West Zone 2', 'west 2': 'West Zone 2',
    'north zone 1': 'North Zone 1', 'north 1': 'North Zone 1',
    'north zone 2': 'North Zone 2', 'north 2': 'North Zone 2',
    'south zone 1': 'South Zone 1', 'south 1': 'South Zone 1',
    'south zone 2': 'South Zone 2', 'south 2': 'South Zone 2',
    'east zone 1': 'East Zone 1', 'east 1': 'East Zone 1',
    'east zone 2': 'East Zone 2', 'east 2': 'East Zone 2',
}

CORRIDOR_KEYWORDS = {
    'mysore road': 'Mysore Road', 'mysore': 'Mysore Road',
    'bellary road 1': 'Bellary Road 1', 'bellary 1': 'Bellary Road 1',
    'bellary road 2': 'Bellary Road 2', 'bellary 2': 'Bellary Road 2',
    'bellary': 'Bellary Road 1',
    'tumkur road': 'Tumkur Road', 'tumkur': 'Tumkur Road',
    'hosur road': 'Hosur Road', 'hosur': 'Hosur Road',
    'orr north': 'ORR North 1', 'orr north 1': 'ORR North 1',
    'orr east': 'ORR East 1', 'orr east 1': 'ORR East 1',
    'old madras road': 'Old Madras Road', 'old madras': 'Old Madras Road',
    'magadi road': 'Magadi Road', 'magadi': 'Magadi Road',
}

DAY_KEYWORDS = {
    'monday': 0, 'mon': 0,
    'tuesday': 1, 'tue': 1, 'tues': 1,
    'wednesday': 2, 'wed': 2,
    'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
    'friday': 4, 'fri': 4,
    'saturday': 5, 'sat': 5,
    'sunday': 6, 'sun': 6,
}


# ─── DATA CLASSES ───────────────────────────────────────────────
@dataclass
class ParsedIntent:
    intent: str              # predict | hotspot | compare | explain | status | help
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # rule-based confidence (0-1)


# ─── INTENT PARSER ──────────────────────────────────────────────
def parse_user_message(text: str) -> ParsedIntent:
    """
    Parse a natural-language message into a structured intent + parameters.
    Uses keyword/regex matching — no external LLM API required.
    """
    text_lower = text.lower().strip()

    # ── Greeting / Help ──────────────────────────────────────────
    if text_lower in ('hi', 'hello', 'hey', 'help', '?', 'what can you do',
                       'what can you do?', 'menu', 'commands'):
        return ParsedIntent(intent='help', params={})

    # ── Status / Metrics ─────────────────────────────────────────
    if any(w in text_lower for w in ['accuracy', 'performance', 'metrics',
                                       'f1', 'auc', 'mae', 'r2', 'r²',
                                       'how good', 'how accurate',
                                       'model score', 'model performance']):
        return ParsedIntent(intent='status', params={})

    # ── Compare intent ───────────────────────────────────────────
    if 'compare' in text_lower or ' vs ' in text_lower or 'versus' in text_lower:
        corridors = []
        for kw, canonical in sorted(CORRIDOR_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in text_lower and canonical not in corridors:
                corridors.append(canonical)
        zones = []
        for kw, canonical in sorted(ZONE_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in text_lower and canonical not in zones:
                zones.append(canonical)
        return ParsedIntent(intent='compare',
                            params={'corridors': corridors, 'zones': zones})

    # ── Hotspot intent ───────────────────────────────────────────
    if any(w in text_lower for w in ['hotspot', 'cluster', 'dense area',
                                       'dangerous area', 'danger zone',
                                       'high risk area', 'worst area',
                                       'most incidents']):
        zone = None
        for kw, canonical in sorted(ZONE_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in text_lower:
                zone = canonical
                break
        return ParsedIntent(intent='hotspot', params={'zone': zone})

    # ── Explain intent ───────────────────────────────────────────
    if any(w in text_lower for w in ['why', 'explain', 'how does',
                                       'what does', 'how do you',
                                       'how is', 'what is the logic',
                                       'reasoning', 'how are recommendations']):
        return ParsedIntent(intent='explain', params={'query': text})

    # ── Predict intent (most complex — extract all params) ───────
    # Try to detect lat/lon
    lat_lon = re.findall(r'(\d{2}\.\d{2,6})\s*[,\s]\s*(\d{2}\.\d{2,6})', text)

    # Try to detect hour
    hour = None
    hour_match = re.search(r'(\d{1,2})\s*(am|pm)', text_lower)
    if hour_match:
        h = int(hour_match.group(1))
        ampm = hour_match.group(2)
        if ampm == 'pm' and h < 12:
            h += 12
        elif ampm == 'am' and h == 12:
            h = 0
        hour = h
    else:
        hour24_match = re.search(r'(\d{1,2})\s*(?::00|hours?|hrs?|o\'?clock)', text_lower)
        if hour24_match:
            hour = int(hour24_match.group(1))
        else:
            # "at 17" or "around 8"
            at_match = re.search(r'(?:at|around|near)\s+(\d{1,2})(?:\s|$|,)', text_lower)
            if at_match:
                hour = int(at_match.group(1))

    # Detect event cause
    detected_cause = None
    for kw, canonical in sorted(CAUSE_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if kw in text_lower:
            detected_cause = canonical
            break

    # Detect zone
    detected_zone = None
    for kw, canonical in sorted(ZONE_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if kw in text_lower:
            detected_zone = canonical
            break

    # Detect corridor
    detected_corridor = None
    for kw, canonical in sorted(CORRIDOR_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if kw in text_lower:
            detected_corridor = canonical
            break

    # Detect day of week
    detected_day = None
    for kw, day_num in DAY_KEYWORDS.items():
        if kw in text_lower:
            detected_day = day_num
            break

    # If we have enough signal to predict
    if lat_lon or detected_cause or hour is not None:
        params = {}
        if lat_lon:
            params['latitude'] = float(lat_lon[0][0])
            params['longitude'] = float(lat_lon[0][1])
        if hour is not None:
            params['hour'] = min(max(hour, 0), 23)
        if detected_cause:
            params['event_cause'] = detected_cause
        if detected_zone:
            params['zone'] = detected_zone
        if detected_corridor:
            params['corridor'] = detected_corridor
        if detected_day is not None:
            params['day_of_week'] = detected_day
        return ParsedIntent(intent='predict', params=params)

    # ── Fallback ─────────────────────────────────────────────────
    return ParsedIntent(intent='help', params={'fallback': True})


# ─── RESPONSE GENERATORS ───────────────────────────────────────
def generate_help_response(params: dict) -> str:
    """Return a help message listing chatbot capabilities."""
    fallback_note = ""
    if params.get('fallback'):
        fallback_note = "🤔 I didn't quite understand that. Here's what I can do:\n\n"

    return f"""{fallback_note}**🤖 ASTRAM Assistant — What I Can Do:**

🔮 **Predict congestion impact** — Give me event details and I'll predict priority, road closure risk, duration, and recommend resources.
> *"accident at 12.97, 77.59 at 5pm"*
> *"waterlogging near Central Zone 2 at 8am on Monday"*
> *"predict breakdown on Hosur Road at 9am"*

🗺️ **Show hotspots** — I'll show you the most congestion-dense areas.
> *"show hotspots"* or *"hotspots in North Zone 1"*

⚖️ **Compare corridors/zones** — Compare risk levels between corridors.
> *"compare Mysore Road vs Hosur Road"*

❓ **Explain recommendations** — I'll explain how the system makes decisions.
> *"why is priority high for protests?"*
> *"how are recommendations generated?"*

📊 **Model performance** — Get current model accuracy metrics.
> *"model accuracy"* or *"how good are the predictions?"*
"""


def generate_predict_response(params: dict, models: dict, encode_geohash_fn,
                                generate_recommendations_fn) -> str:
    """Run ML models and return prediction + recommendations as formatted text."""
    from datetime import datetime

    # Defaults for missing params
    latitude = params.get('latitude', 12.9716)
    longitude = params.get('longitude', 77.5946)
    event_hour = params.get('hour', datetime.now().hour)
    event_cause = params.get('event_cause', 'congestion')
    zone = params.get('zone', 'Central Zone 2')
    corridor = params.get('corridor', 'Non-corridor')
    day_of_week = params.get('day_of_week', datetime.now().weekday())
    month = params.get('month', datetime.now().month)
    veh_type = params.get('veh_type', 'others')
    police_station = params.get('police_station', 'Cubbon Park')

    # ── Build feature vector (same logic as app.py predictor page) ──
    hour_bins = [-1, 5, 11, 16, 20, 24]
    hour_labels = ['night', 'morning', 'afternoon', 'evening', 'late_evening']
    tod = pd.cut([event_hour], bins=hour_bins, labels=hour_labels)[0]

    geohash_lookup = models['geohash_lookup']
    zone_hour_lookup = models['zone_hour_lookup']
    corridor_risk_lookup = models['corridor_risk_lookup']
    global_medians = models['global_medians']
    closure_best_thresh = models['closure_best_thresh']

    geohash_val = encode_geohash_fn(latitude, longitude, precision=6)

    # Geo stats lookup
    if geohash_val in geohash_lookup:
        g = geohash_lookup[geohash_val]
        geo_event_count = g.get('geo_event_count', global_medians['geo_event_count'])
        geo_high_priority_rate = g.get('geo_high_priority_rate', global_medians['geo_high_priority_rate'])
        geo_closure_rate = g.get('geo_closure_rate', global_medians['geo_closure_rate'])
        geo_avg_duration = g.get('geo_avg_duration', global_medians['geo_avg_duration'])
    else:
        geo_event_count = global_medians['geo_event_count']
        geo_high_priority_rate = global_medians['geo_high_priority_rate']
        geo_closure_rate = global_medians['geo_closure_rate']
        geo_avg_duration = global_medians['geo_avg_duration']

    # Zone-hour lookup
    zh_key = (zone, event_hour)
    if zh_key in zone_hour_lookup:
        zone_hour_event_count = zone_hour_lookup[zh_key].get(
            'zone_hour_event_count', global_medians['zone_hour_event_count'])
    else:
        zone_hour_event_count = global_medians['zone_hour_event_count']

    # Corridor risk lookup
    corridor_key = corridor.strip().lower()
    if corridor_key in corridor_risk_lookup:
        corridor_risk_score = corridor_risk_lookup[corridor_key].get(
            'corridor_risk_score', global_medians['corridor_risk_score'])
    else:
        corridor_risk_score = global_medians['corridor_risk_score']

    # Cause closure rate lookup (RC-1)
    if event_cause in models['cause_closure_lookup']:
        cause_closure_rate = models['cause_closure_lookup'][event_cause]
    else:
        cause_closure_rate = global_medians['cause_closure_rate']

    # Cause average duration lookup (DR-1)
    if event_cause in models['cause_duration_lookup']:
        cause_avg_duration = models['cause_duration_lookup'][event_cause]
    else:
        cause_avg_duration = global_medians['cause_avg_duration']

    input_dict = {
        'is_planned': 0,
        'hour': event_hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': int(day_of_week >= 5),
        'is_peak_hour': int(event_hour in [7, 8, 9, 17, 18, 19, 20]),
        'latitude': latitude,
        'longitude': longitude,
        'event_cause': event_cause,
        'veh_type': veh_type,
        'corridor': corridor,
        'police_station': police_station,
        'zone': zone,
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

    # Add time_of_day OHE dummies
    for dummy_col in models['preprocessor']['time_dummies']:
        label = dummy_col.replace('time_of_day_', '')
        input_dict[dummy_col] = int(str(tod) == label)

    input_df_base = pd.DataFrame([input_dict])

    def align_features(base_df, feat_list):
        df_aligned = base_df.copy()
        for c in feat_list:
            if c not in df_aligned.columns:
                df_aligned[c] = 0
        return df_aligned[feat_list]

    input_df_priority = align_features(input_df_base, models['features_priority'])
    input_df_closure  = align_features(input_df_base, models['features_closure'])
    input_df_all      = align_features(input_df_base, models['features'])

    # Run predictions
    prio_prob  = models['priority'].predict_proba(input_df_priority)[0][1]
    close_prob = models['closure'].predict_proba(input_df_closure)[0][1]
    try:
        dur_log = models['duration'].predict(input_df_all)[0]
        dur_est = max(1.0, np.expm1(dur_log))
    except Exception as e:
        logging.warning(f"Duration prediction failed: {e}")
        dur_est = 60.0

    recs = generate_recommendations_fn(
        prio_prob, close_prob, dur_est, event_cause, event_hour, zone,
        closure_thresh=closure_best_thresh
    )

    # ── Format response ──
    prio_icon = "🔴" if prio_prob > 0.5 else "🟢"
    closure_icon = "🔴" if close_prob >= closure_best_thresh else "🟢"

    # Show which params were auto-filled
    provided_keys = set(params.keys())
    defaulted = []
    if 'latitude' not in provided_keys:
        defaulted.append("location (default: Majestic)")
    if 'hour' not in provided_keys:
        defaulted.append(f"hour (default: {event_hour}h — current)")
    if 'event_cause' not in provided_keys:
        defaulted.append("cause (default: congestion)")
    if 'zone' not in provided_keys:
        defaulted.append("zone (default: Central Zone 2)")

    defaults_note = ""
    if defaulted:
        defaults_note = f"\n\n> ℹ️ *Auto-filled defaults: {', '.join(defaulted)}. Mention them in your query for more accurate predictions.*"

    return f"""**🔮 Congestion Impact Prediction**

| Metric | Value |
|---|---|
| {prio_icon} **Priority Risk** | **{prio_prob*100:.1f}%** ({'HIGH' if prio_prob > 0.5 else 'LOW'}) |
| {closure_icon} **Road Closure Risk** | **{close_prob*100:.1f}%** ({'LIKELY' if close_prob >= closure_best_thresh else 'UNLIKELY'}) |
| ⏱️ **Estimated Duration** | **{recs['estimated_duration_min']} min** |
| 📍 **Location** | {latitude:.4f}, {longitude:.4f} (Geohash: `{geohash_val}`) |
| 🕐 **Time Context** | {event_hour}:00, {'Weekend' if day_of_week >= 5 else 'Weekday'}, {'Peak' if event_hour in [7,8,9,17,18,19,20] else 'Off-peak'} |

**👮 Recommended Response:**

| Resource | Recommendation |
|---|---|
| 👮 **Manpower** | {recs['manpower']} |
| 🚧 **Barricading** | {recs['barricading']} |
| 🔀 **Diversion** | {recs['diversion']} |
{defaults_note}"""


def generate_hotspot_response(params: dict, models: dict) -> str:
    """Return hotspot cluster information, optionally filtered by zone."""
    cp = models['cluster_profile']
    pts = models['cluster_points']
    zone_filter = params.get('zone')

    if zone_filter:
        # Filter cluster points to those in the specified zone
        # Cluster profiles don't have zone — we match by looking at points
        if 'zone' in pts.columns:
            zone_pts = pts[(pts['zone'] == zone_filter) & (pts['cluster'] >= 0)]
            if zone_pts.empty:
                return f"📍 No hotspot clusters found in **{zone_filter}**. Try a broader query like *'show all hotspots'*."
            relevant_clusters = zone_pts['cluster'].unique()
            cp_filtered = cp[cp['cluster'].isin(relevant_clusters)]
        else:
            cp_filtered = cp
    else:
        cp_filtered = cp

    if cp_filtered.empty:
        return "No hotspot clusters found in the dataset."

    top5 = cp_filtered.nlargest(5, 'event_count')

    rows = ""
    for _, row in top5.iterrows():
        rows += f"| {int(row['cluster'])} | {int(row['event_count'])} | {row['high_priority_pct']:.1f}% | {row['centroid_lat']:.4f}, {row['centroid_lon']:.4f} | {row['top_cause']} |\n"

    zone_label = f" in **{zone_filter}**" if zone_filter else ""
    total = len(cp)
    return f"""**🗺️ Top Congestion Hotspots{zone_label}**

Showing top {len(top5)} of {total} DBSCAN clusters (sorted by incident count):

| Cluster | Events | High Priority % | Centroid | Top Cause |
|---|---|---|---|---|
{rows}
> 💡 *Visit the **🗺️ Hotspot Map** page for an interactive map visualization.*"""


def generate_compare_response(params: dict, models: dict) -> str:
    """Compare corridor risk scores or zone statistics."""
    corridors = params.get('corridors', [])
    zones = params.get('zones', [])
    corridor_risk_lookup = models['corridor_risk_lookup']

    results = []

    if corridors:
        rows = ""
        for c in corridors:
            key = c.strip().lower()
            if key in corridor_risk_lookup:
                risk = corridor_risk_lookup[key].get('corridor_risk_score', 0)
                risk_label = "🔴 HIGH" if risk > 0.7 else ("🟡 MEDIUM" if risk > 0.4 else "🟢 LOW")
                rows += f"| {c} | {risk*100:.1f}% | {risk_label} |\n"
            else:
                rows += f"| {c} | N/A | Not found in lookup |\n"
        results.append(f"""**⚖️ Corridor Risk Comparison**

| Corridor | High-Priority Rate | Risk Level |
|---|---|---|
{rows}""")

    if zones:
        zone_hour_lookup = models['zone_hour_lookup']
        rows = ""
        for z in zones:
            # Sum events across all hours for this zone
            total = sum(v.get('zone_hour_event_count', 0)
                        for k, v in zone_hour_lookup.items()
                        if k[0] == z)
            peak = max((v.get('zone_hour_event_count', 0)
                        for k, v in zone_hour_lookup.items()
                        if k[0] == z), default=0)
            rows += f"| {z} | {total} | {peak} |\n"
        results.append(f"""**📊 Zone Comparison**

| Zone | Total Events (all hours) | Peak Hour Events |
|---|---|---|
{rows}""")

    if not results:
        return ("I couldn't find corridors or zones to compare. "
                "Try: *'compare Mysore Road vs Hosur Road'* or *'compare Central Zone 1 vs West Zone 2'*")

    return "\n\n".join(results)


def generate_explain_response(params: dict) -> str:
    """Explain how the recommendation system works."""
    query = params.get('query', '').lower()

    if any(w in query for w in ['priority', 'high', 'low']):
        return """**❓ How Priority Predictions Work**

The **LightGBM classifier** predicts whether an event will be classified as High or Low priority based on:

1. **Geospatial context** — Historical incident density and severity in the geohash cell
2. **Temporal patterns** — Hour of day, peak hours, weekday vs weekend
3. **Event characteristics** — Cause type, vehicle type, planned vs unplanned
4. **Zone pressure** — How many events typically occur in that zone at that hour

The model outputs a probability (0–100%). Above 50% → **HIGH** priority.

> ⚠️ **Special override**: Protests, processions, public events, and VIP movements are **always** classified HIGH regardless of model output — for public safety."""

    if any(w in query for w in ['closure', 'road closure', 'close']):
        return """**❓ How Road Closure Predictions Work**

The **XGBoost classifier** predicts whether a road closure will be required:

1. Road closures are rare (~8.2% of events), so the model uses **SMOTE oversampling** to learn the minority pattern
2. A **custom threshold** (tuned on validation data) is used instead of the default 0.5
3. The threshold favors **recall** — it's better to over-predict closures than miss one

Risk levels:
- **LIKELY** (above threshold): Deploy heavy barricades + route advisory
- **UNLIKELY** (below threshold): Cones only"""

    if any(w in query for w in ['duration', 'time', 'how long', 'minutes']):
        return """**❓ How Duration Predictions Work**

The **LightGBM Regressor** predicts how long a congestion event will last:

1. Trained on `log1p(duration_minutes)` to handle skewed distribution
2. Predictions are back-transformed via `expm1()` to get minutes
3. **Limitation**: Only ~6% of events have a recorded end time, so this model has MAE ≈ 53 min

> 💡 Duration predictions should be treated as **rough estimates**, not precise forecasts."""

    if any(w in query for w in ['recommend', 'manpower', 'barricad', 'diversion']):
        return """**❓ How Recommendations Are Generated**

The recommendation engine is a **rule-based layer** on top of ML predictions:

**👮 Manpower:**
| Priority Risk | Recommendation |
|---|---|
| ≥ 75% | HIGH — 8–12 officers + 2 PCR vans |
| 45–75% | MEDIUM — 4–6 officers + 1 PCR van |
| < 45% | LOW — 2 officers |

**🚧 Barricading:**
| Closure Probability | Action |
|---|---|
| ≥ threshold | FULL — Heavy barricades + signage |
| ≥ threshold × 0.5 | PARTIAL — Lane-level barricading |
| Below | MINIMAL — Cones only |

**🔀 Diversion:**
- Closure likely + peak hour → **URGENT** (activate alternate routes + Waze/Google Maps notification)
- Moderate risk or duration > 90 min → **RECOMMENDED** (pre-position advisory boards)
- Otherwise → **MONITOR**"""

    # Generic explanation
    return """**❓ How ASTRAM Predictions Work**

The system uses **3 ML models** working together:

1. **Priority Classifier** (LightGBM) — Predicts if an event is High/Low priority
2. **Road Closure Classifier** (XGBoost) — Predicts if a road closure is needed
3. **Duration Regressor** (LightGBM) — Estimates how long the event will last

These predictions feed into a **rule-based recommendation engine** that suggests:
- 👮 Manpower deployment levels
- 🚧 Barricading requirements
- 🔀 Diversion urgency

Ask me more specifically about any of these: *"explain priority"*, *"how does closure prediction work?"*, *"how are recommendations generated?"*"""


def generate_status_response(models: dict) -> str:
    """Return model evaluation metrics summary."""
    metrics = models.get('eval_metrics', {})

    return f"""**📊 Current Model Performance Metrics**

| Model | Metric | Value |
|---|---|---|
| ⚡ Priority (LightGBM) | ROC-AUC | **{metrics.get('priority_roc_auc', 'N/A')}** |
| ⚡ Priority (LightGBM) | Weighted F1 (CV) | **{metrics.get('priority_f1_weighted', 'N/A')}** |
| 🚧 Road Closure (XGBoost) | ROC-AUC | **{metrics.get('closure_roc_auc', 'N/A')}** |
| 🚧 Road Closure (XGBoost) | Best Threshold | **{metrics.get('closure_best_threshold', 'N/A')}** |
| ⏱️ Duration (LGBMRegressor) | MAE | **{metrics.get('duration_mae', 'N/A')} min** |
| ⏱️ Duration (LGBMRegressor) | Median AE | **{metrics.get('duration_medae', 'N/A')} min** |
| ⏱️ Duration (LGBMRegressor) | R² Score | **{metrics.get('duration_r2', 'N/A')}** |

> 📋 *Visit the **📋 Model Performance** page for detailed reports and feature importance charts.*"""


# ─── MAIN DISPATCHER ────────────────────────────────────────────
def execute_intent(parsed: ParsedIntent, models: dict,
                   encode_geohash_fn=None,
                   generate_recommendations_fn=None) -> str:
    """
    Route a parsed intent to the appropriate response generator.

    Parameters
    ----------
    parsed : ParsedIntent
        Output of parse_user_message()
    models : dict
        The loaded model dictionary from app.py
    encode_geohash_fn : callable
        The encode_geohash() function from app.py
    generate_recommendations_fn : callable
        The generate_recommendations() function from app.py
    """
    try:
        if parsed.intent == 'help':
            return generate_help_response(parsed.params)

        elif parsed.intent == 'predict':
            if encode_geohash_fn is None or generate_recommendations_fn is None:
                return "⚠️ Prediction functions not available. Please check model loading."
            return generate_predict_response(
                parsed.params, models,
                encode_geohash_fn, generate_recommendations_fn)

        elif parsed.intent == 'hotspot':
            return generate_hotspot_response(parsed.params, models)

        elif parsed.intent == 'compare':
            return generate_compare_response(parsed.params, models)

        elif parsed.intent == 'explain':
            return generate_explain_response(parsed.params)

        elif parsed.intent == 'status':
            return generate_status_response(models)

        else:
            return generate_help_response({'fallback': True})

    except Exception as e:
        return f"⚠️ An error occurred while processing your request:\n\n`{str(e)}`\n\nPlease try rephrasing your question."
