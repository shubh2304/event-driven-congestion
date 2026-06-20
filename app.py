import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Import Chatbot & Online Learning Modules ──────────────────
from chatbot import parse_user_message, execute_intent
from online_update import PredictionLogger, DriftDetector

# Initialize prediction logger
prediction_logger = PredictionLogger()

# ─── GEOCONVERTER FALLBACK ────────────────────────────────────
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

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ASTRAM — Bengaluru Event Congestion Forecaster",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS Styling ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Glassmorphic Metric Cards */
.custom-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}

.custom-card:hover {
    transform: translateY(-4px);
    border-color: rgba(255,255,255,0.25);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.35);
}

.card-title {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.card-value {
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 0.5rem;
    background: linear-gradient(90deg, #ffffff, #cfd9df);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.badge-high {
    background-color: rgba(239, 68, 68, 0.2);
    color: rgb(239, 68, 68);
    border: 1px solid rgb(239, 68, 68);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}

.badge-low {
    background-color: rgba(16, 185, 129, 0.2);
    color: rgb(16, 185, 129);
    border: 1px solid rgb(16, 185, 129);
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
}

.rec-container {
    border-left: 4px solid #3b82f6;
    padding-left: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Models & Lookups ─────────────────────────────────────
@st.cache_resource
def load_models():
    import json
    return {
        'priority':          joblib.load('models/priority_classifier.pkl'),
        'closure':           joblib.load('models/closure_classifier.pkl'),
        'duration':          joblib.load('models/duration_regressor.pkl'),
        'cluster_profile':   joblib.load('models/cluster_profiles.pkl'),
        'cluster_points':    joblib.load('models/cluster_points.pkl'),
        'features_priority': joblib.load('models/feature_list_priority.pkl'),  # priority model
        'features_closure':  joblib.load('models/feature_list_closure.pkl'),   # closure model
        'features':          joblib.load('models/feature_list.pkl'),            # duration model
        'geohash_lookup':    joblib.load('models/geohash_lookup.pkl'),
        'zone_hour_lookup':  joblib.load('models/zone_hour_lookup.pkl'),
        'corridor_risk_lookup': joblib.load('models/corridor_risk_lookup.pkl'),
        'cause_closure_lookup': joblib.load('models/cause_closure_lookup.pkl'),
        'cause_duration_lookup': joblib.load('models/cause_duration_lookup.pkl'),
        'global_medians':    joblib.load('models/global_medians.pkl'),
        'closure_best_thresh': joblib.load('models/closure_best_threshold.pkl'),
        'preprocessor':      joblib.load('models/preprocessor.pkl'),            # IQR bounds + mode values + time_dummies
        'eval_metrics':      json.load(open('models/eval_metrics.json')),
    }

# Safe model load
models_loaded = False
try:
    models = load_models()
    models_loaded = True
except Exception as e:
    models_error = str(e)

# ─── Recommendation Engine ─────────────────────────────────────
def generate_recommendations(priority_prob: float, closure_prob: float, 
                               duration_est: float, event_cause: str,
                               hour: int, zone: str, closure_thresh: float = 0.396) -> dict:
    """
    Rule-based recommendation layer on top of ML predictions.
    Returns manpower, barricading level, and diversion urgency.
    """
    # Manpower
    if priority_prob >= 0.75:
        manpower = "HIGH — Deploy 8–12 officers + 2 PCR vans"
    elif priority_prob >= 0.45:
        manpower = "MEDIUM — Deploy 4–6 officers + 1 PCR van"
    else:
        manpower = "LOW — 2 officers sufficient"

    # Barricading
    if closure_prob >= closure_thresh:
        barricading = "FULL ROAD CLOSURE — Deploy heavy barricades, signage, and rerouting boards"
    elif closure_prob >= closure_thresh * 0.5:
        barricading = "PARTIAL — Lane-level barricading recommended"
    else:
        barricading = "MINIMAL — Cones / soft barricades only"

    # Diversion
    is_peak = hour in [7,8,9,17,18,19,20]
    if closure_prob >= closure_thresh and is_peak:
        diversion = "URGENT — Activate alternate route NOW. Notify Waze/Google Maps."
    elif closure_prob >= closure_thresh * 0.75 or duration_est > 90:
        diversion = "RECOMMENDED — Pre-position route advisory boards"
    else:
        diversion = "MONITOR — No diversion needed currently"

    # Special cause overrides
    if event_cause in ['public_event', 'procession', 'protest', 'vip_movement']:
        manpower = "HIGH — Political/public event protocol. Coordinate with event organizers."
        barricading = "FULL ROAD CLOSURE recommended for public safety"

    return {
        'manpower': manpower,
        'barricading': barricading,
        'diversion': diversion,
        'estimated_duration_min': round(duration_est),
        'priority_score': f"{priority_prob*100:.1f}%",
        'closure_risk': f"{closure_prob*100:.1f}%"
    }

# ─── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("🚦 ASTRAM Congestion Forecaster")
page = st.sidebar.radio("Navigate", [
    "🔮 Event Impact Predictor",
    "🗺️ Hotspot Map",
    "📊 Dataset Analytics",
    "📋 Model Performance",
    "💬 ASTRAM Assistant",
    "📡 Model Monitoring"
])

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Bengaluru Traffic Management**\n\n"
    "Using the ASTRAM dataset to run live predictions, identify geospatial hotspots, "
    "and recommend optimal field resources."
)

if not models_loaded:
    st.error("⚠️ Model files not found in the `models/` directory!")
    st.warning("Please run the ML training pipeline first to train and serialize the models:")
    st.code("python train_pipeline.py", language="bash")
    st.info(f"System Load Error: {models_error}")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTOR
# ═══════════════════════════════════════════════════════════════
if page == "🔮 Event Impact Predictor":
    st.title("🔮 Event Impact Predictor & Recommendation Engine")
    st.markdown("Forecast traffic congestion impact in real time and get actionable deployment recommendations.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📝 Event Details")
        event_type = st.selectbox("Event Type", ["unplanned", "planned"])
        event_cause = st.selectbox("Event Cause", [
            "vehicle_breakdown", "accident", "construction", "pot_holes",
            "water_logging", "public_event", "procession", "vip_movement",
            "protest", "tree_fall", "road_conditions", "congestion",
            "fog_low_visibility", "others"
        ])
        veh_type = st.selectbox("Vehicle Type Involved", [
            "others", "bmtc_bus", "heavy_vehicle", "lcv", "truck", "private_bus",
            "private_car", "ksrtc_bus", "taxi", "auto"
        ])

    with col2:
        st.subheader("📍 Geospatial Context")
        zone = st.selectbox("Bengaluru Zone", [
            "Central Zone 2", "West Zone 1", "North Zone 2", "West Zone 2",
            "South Zone 2", "North Zone 1", "Central Zone 1", "East Zone 1",
            "South Zone 1", "East Zone 2"
        ])
        corridor = st.selectbox("Corridor Name", [
            "Non-corridor", "Mysore Road", "Bellary Road 1", "Tumkur Road",
            "Bellary Road 2", "Hosur Road", "ORR North 1", "Old Madras Road",
            "Magadi Road", "ORR East 1", "Other"
        ])
        police_station = st.text_input("Jurisdiction Police Station", "Cubbon Park")

    with col3:
        st.subheader("🕒 Location & Timing")
        latitude = st.number_input("Latitude", value=12.9716, format="%.6f",
                                    min_value=12.75, max_value=13.30)
        longitude = st.number_input("Longitude", value=77.5946, format="%.6f",
                                     min_value=77.25, max_value=77.85)
        event_hour = st.slider("Hour of Day", 0, 23, 8)
        event_day = st.selectbox("Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        event_month = st.slider("Month of Year", 1, 12, 6)

    predict_btn = st.button("🔮 Predict & Recommend", type="primary", use_container_width=True)

    if predict_btn:
        # Day of week mapping
        day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                   "Friday":4,"Saturday":5,"Sunday":6}
        hour_bins = [-1,5,11,16,20,24]
        hour_labels = ['night','morning','afternoon','evening','late_evening']
        tod = pd.cut([event_hour], bins=hour_bins, labels=hour_labels)[0]

        # ─── Production Lookup Integration ───
        geohash_lookup = models['geohash_lookup']
        zone_hour_lookup = models['zone_hour_lookup']
        corridor_risk_lookup = models['corridor_risk_lookup']
        global_medians = models['global_medians']
        closure_best_thresh = models['closure_best_thresh']

        geohash_val = encode_geohash(latitude, longitude, precision=6)
        
        # Real lookup for geo stats
        if geohash_val in geohash_lookup:
            g_stats = geohash_lookup[geohash_val]
            geo_event_count = g_stats.get('geo_event_count', global_medians['geo_event_count'])
            geo_high_priority_rate = g_stats.get('geo_high_priority_rate', global_medians['geo_high_priority_rate'])
            geo_closure_rate = g_stats.get('geo_closure_rate', global_medians['geo_closure_rate'])
            geo_avg_duration = g_stats.get('geo_avg_duration', global_medians['geo_avg_duration'])
        else:
            geo_event_count = global_medians['geo_event_count']
            geo_high_priority_rate = global_medians['geo_high_priority_rate']
            geo_closure_rate = global_medians['geo_closure_rate']
            geo_avg_duration = global_medians['geo_avg_duration']

        # Real lookup for zone x hour
        zh_key = (zone, event_hour)
        if zh_key in zone_hour_lookup:
            zone_hour_event_count = zone_hour_lookup[zh_key].get('zone_hour_event_count', global_medians['zone_hour_event_count'])
        else:
            zone_hour_event_count = global_medians['zone_hour_event_count']

        # Real lookup for corridor risk
        corridor_key = corridor.strip().lower()
        if corridor_key in corridor_risk_lookup:
            corridor_risk_score = corridor_risk_lookup[corridor_key].get('corridor_risk_score', global_medians['corridor_risk_score'])
        else:
            corridor_risk_score = global_medians['corridor_risk_score']

        # Real lookup for cause closure rate
        if event_cause in models['cause_closure_lookup']:
            cause_closure_rate = models['cause_closure_lookup'][event_cause]
        else:
            cause_closure_rate = global_medians['cause_closure_rate']

        # Real lookup for cause average duration
        if event_cause in models['cause_duration_lookup']:
            cause_avg_duration = models['cause_duration_lookup'][event_cause]
        else:
            cause_avg_duration = global_medians['cause_avg_duration']

        input_dict = {
            'is_planned': int(event_type == 'planned'),
            'hour': event_hour,
            'day_of_week': day_map[event_day],
            'month': event_month,
            'is_weekend': int(day_map[event_day] >= 5),
            'is_peak_hour': int(event_hour in [7,8,9,17,18,19,20]),
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

        # Add time_of_day OHE dummies using the exact column list saved at training time.
        # This replaces a hardcoded ['morning','afternoon','evening','late_evening'] list
        # that would silently break if bins or label order ever changed.
        for dummy_col in models['preprocessor']['time_dummies']:
            label = dummy_col.replace('time_of_day_', '')
            input_dict[dummy_col] = int(str(tod) == label)

        input_df_base = pd.DataFrame([input_dict])

        def align_features(base_df, feat_list):
            """Align inference row to a specific feature list, zero-filling missing cols."""
            df_aligned = base_df.copy()
            for c in feat_list:
                if c not in df_aligned.columns:
                    df_aligned[c] = 0
            return df_aligned[feat_list]

        # Priority model uses FEATURES_PRIORITY
        input_df_priority = align_features(input_df_base, models['features_priority'])
        # Closure model uses FEATURES_CLOSURE
        input_df_closure  = align_features(input_df_base, models['features_closure'])
        # Duration model uses FEATURES_ALL
        input_df_all      = align_features(input_df_base, models['features'])

        with st.spinner("Running ML models..."):
            prio_prob  = models['priority'].predict_proba(input_df_priority)[0][1]
            close_prob = models['closure'].predict_proba(input_df_closure)[0][1]
            try:
                dur_log = models['duration'].predict(input_df_all)[0]
                dur_est = max(1.0, np.expm1(dur_log))
            except Exception as e:
                logging.warning(f"Duration prediction failed: {e}")
                dur_est = 60.0

        recs = generate_recommendations(prio_prob, close_prob, dur_est, event_cause, event_hour, zone, closure_thresh=closure_best_thresh)

        # ─── Log Prediction ───
        prediction_logger.log_prediction(
            input_params=input_dict,
            predictions={'priority_prob': prio_prob, 'closure_prob': close_prob,
                         'duration_est': dur_est},
            recommendations=recs
        )

        # ─── Results Layout ───
        st.markdown("---")
        st.subheader("📊 Model Predictions & Impact Score")
        
        # Top Metric Cards Row
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            prio_badge = f'<div class="badge-high">HIGH</div>' if prio_prob > 0.5 else f'<div class="badge-low">LOW</div>'
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">⚡ Priority Risk</div>
                <div class="card-value">{recs['priority_score']}</div>
                <div style="margin-top: 10px;">{prio_badge}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with r2:
            closure_badge = f'<div class="badge-high">Likely</div>' if close_prob >= closure_best_thresh else f'<div class="badge-low">Unlikely</div>'
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">🚧 Road Closure Risk</div>
                <div class="card-value">{recs['closure_risk']}</div>
                <div style="margin-top: 10px;">{closure_badge} (Thresh: {closure_best_thresh:.2f})</div>
            </div>
            """, unsafe_allow_html=True)
            
        with r3:
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">⏱️ Est. Duration</div>
                <div class="card-value">{recs['estimated_duration_min']} min</div>
                <div style="margin-top: 10px; font-size: 0.85rem; color: rgba(255,255,255,0.4)">Log-transformed LGBM Regressor</div>
            </div>
            """, unsafe_allow_html=True)
            
        with r4:
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-title">📍 Context Zone</div>
                <div class="card-value" style="font-size: 1.5rem;">{zone}</div>
                <div style="margin-top: 10px; font-size: 0.85rem; color: rgba(255,255,255,0.4)">Geohash: <code>{geohash_val}</code></div>
            </div>
            """, unsafe_allow_html=True)

        # Operational Deployment Recommendations
        st.subheader("👮 Operational Recommendations")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info(f"👮 **Manpower:**\n\n{recs['manpower']}")
        with col_b:
            st.warning(f"🚧 **Barricading:**\n\n{recs['barricading']}")
        with col_c:
            st.error(f"🔀 **Diversion Urgency:**\n\n{recs['diversion']}")

        # Gauge chart for priority score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prio_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Priority Risk Percentage", 'font': {'size': 20, 'family': 'Outfit'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ef4444"},
                'bgcolor': "rgba(255, 255, 255, 0.05)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.1)"},
                    {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.1)"},
                    {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.1)"},
                ]
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'family': 'Outfit'},
            margin=dict(t=50, b=10, l=10, r=10),
            height=280
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — HOTSPOT MAP
# ═══════════════════════════════════════════════════════════════
elif page == "🗺️ Hotspot Map":
    st.title("🗺️ Bengaluru Event Hotspot Map")
    st.markdown("Geospatial visualization of DBSCAN-identified congestion hotspot clusters across Bengaluru.")

    cp = models['cluster_profile']
    pts = models['cluster_points']

    # Show Mapbox map
    fig_map = px.scatter_mapbox(
        pts[pts['cluster'] >= 0],
        lat='latitude', lon='longitude',
        color='cluster', size_max=10,
        hover_data=['event_cause', 'priority'],
        color_continuous_scale='Turbo',
        mapbox_style='carto-positron',
        zoom=11, center={"lat": 12.9716, "lon": 77.5946},
        title="Active DBSCAN Hotspot Clusters"
    )
    fig_map.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Outfit'},
        margin=dict(t=40, b=0, l=0, r=0),
        height=600
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("📊 Cluster Profiles")
    st.markdown("Metrics for each density-based cluster (centroids, sizes, high priority rate, and dominant cause):")
    st.dataframe(
        cp.sort_values('event_count', ascending=False).rename(columns={
            'cluster': 'Cluster ID',
            'event_count': 'Total Events',
            'high_priority_pct': 'High Priority (%)',
            'centroid_lat': 'Centroid Lat',
            'centroid_lon': 'Centroid Lon',
            'top_cause': 'Top Primary Cause'
        }),
        use_container_width=True,
        hide_index=True
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — DATASET ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Dataset Analytics":
    st.title("📊 Dataset Analytics Dashboard")
    st.markdown("Exploratory insights of the raw historical ASTRAM congestion database.")

    @st.cache_data
    def load_raw_data():
        if os.path.exists("Astram_event_data_anonymized.csv"):
            df_raw = pd.read_csv("Astram_event_data_anonymized.csv")
        else:
            candidates = [f for f in os.listdir('.') if 'Astram event data_anonymized' in f and f.endswith('.csv')]
            if candidates:
                df_raw = pd.read_csv(candidates[0])
            else:
                return None
        df_raw['start_datetime'] = pd.to_datetime(df_raw['start_datetime'], utc=True, errors='coerce')
        df_raw['hour'] = df_raw['start_datetime'].dt.hour
        return df_raw

    df_raw = load_raw_data()

    if df_raw is None:
        st.error("Could not find the dataset CSV to display analytics. Please make sure the CSV exists.")
    else:
        # Overview stats cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Logged Events", f"{len(df_raw):,}")
        c2.metric("Unplanned Events", f"{(df_raw['event_type']=='unplanned').sum():,}")
        c3.metric("High Priority Events", f"{(df_raw['priority']=='High').sum():,}")
        c4.metric("Requires Road Closures", f"{df_raw['requires_road_closure'].sum():,}")

        col1, col2 = st.columns(2)
        with col1:
            cause_counts = df_raw['event_cause'].str.strip().str.lower().value_counts().reset_index()
            cause_counts.columns = ['Event Cause', 'Count']
            fig1 = px.bar(cause_counts,
                          x='Event Cause', y='Count', title="Distribution of Events by Cause",
                          color='Count', color_continuous_scale='Reds')
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white', 'family': 'Outfit'})
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            hourly = df_raw.groupby('hour').size().reset_index(name='Count')
            fig2 = px.line(hourly, x='hour', y='Count', title="Congestion Peak Hour Distribution",
                           markers=True, line_shape="spline")
            fig2.update_traces(line_color="#ef4444", marker=dict(size=6, color="white"))
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white', 'family': 'Outfit'})
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            zone_data = df_raw.groupby('zone').size().reset_index(name='Count').dropna()
            fig3 = px.pie(zone_data, names='zone', values='Count', title="Share of Events by Bengaluru Zone",
                          hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white', 'family': 'Outfit'})
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            cause_prio = df_raw.groupby(['event_cause', 'priority']).size().reset_index(name='Count')
            fig4 = px.bar(cause_prio, x='event_cause', y='Count', color='priority',
                          title="Priority Casing by Event Cause", barmode='stack',
                          color_discrete_map={'High': '#ef4444', 'Low': '#10b981'})
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white', 'family': 'Outfit'})
            st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Model Performance":
    st.title("📋 Model Performance & Validation Report")
    st.markdown("Post-training classification reports, regression stability scores, and predictor feature importance weights.")

    metrics = models.get('eval_metrics', {})

    t1, t2, t3 = st.tabs(["⚡ Priority Model", "🚧 Road Closure Model", "⏱️ Duration Regressor"])

    with t1:
        st.subheader("Priority Classification (LightGBM)")
        st.markdown("The Priority classifier was trained using target encoding and SMOTE over-sampling. The model is tuned for macro-averaged F1 and general AUC robustness.")
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC", f"{metrics.get('priority_roc_auc', 'N/A')}")
        m2.metric("Weighted F1", f"{metrics.get('priority_f1_weighted', 'N/A')}")
        
    with t2:
        st.subheader("Road Closure Classification (XGBoost)")
        st.markdown("The Road Closure model handles severe class imbalance. SMOTE is used, combined with threshold-tuning on validation predictions to maximize F1-score.")
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC", f"{metrics.get('closure_roc_auc', 'N/A')}")
        m2.metric("Optimal Validation Threshold", f"{metrics.get('closure_best_threshold', 'N/A')}")
        
    with t3:
        st.subheader("Duration Regression (LightGBM)")
        st.markdown("The Duration Regressor predicts the elapsed time of a congestion incident in minutes. It is trained only on records containing a valid end time.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Absolute Error", f"{metrics.get('duration_mae', 'N/A')} min")
        m2.metric("Median Absolute Error", f"{metrics.get('duration_medae', 'N/A')} min")
        m3.metric("R² Score (Original Scale)", f"{metrics.get('duration_r2', 'N/A')}")
        st.caption("ℹ️ *Note: The target duration contains high missing values in ASTRAM. The predictions are evaluated on back-transformed original values (minutes).*")

    # Feature Importance Chart
    st.subheader("📈 Top Predictor Features (Feature Importance)")
    fi_records = metrics.get('priority_feature_importance', [])
    df_fi = pd.DataFrame(fi_records).sort_values('importance', ascending=True)
    fig_fi = px.bar(df_fi, x='importance', y='feature',
                    orientation='h', color='importance',
                    color_continuous_scale='Reds', title="LightGBM Feature Importance (Top 10)")
    fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white', 'family': 'Outfit'})
    st.plotly_chart(fig_fi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — CHATBOT ASSISTANT
# ═══════════════════════════════════════════════════════════════
elif page == "💬 ASTRAM Assistant":
    st.title("💬 ASTRAM Traffic Assistant")
    st.markdown(
        "Ask questions in natural language to get predictions, explore hotspots, "
        "compare corridors, and understand how the models work."
    )

    # Custom chat styling
    st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 I'm the **ASTRAM Traffic Assistant**. I can help you:\n\n"
                    "🔮 **Predict** — *\"accident at 12.97, 77.59 at 5pm\"*\n\n"
                    "🗺️ **Hotspots** — *\"show hotspots in North Zone 1\"*\n\n"
                    "⚖️ **Compare** — *\"compare Mysore Road vs Hosur Road\"*\n\n"
                    "❓ **Explain** — *\"how are recommendations generated?\"*\n\n"
                    "📊 **Metrics** — *\"model accuracy\"*\n\n"
                    "Type your question below to get started! 👇"
                )
            }
        ]

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about traffic events, hotspots, or predictions..."):
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Parse intent and generate response
        with st.spinner("Thinking..."):
            parsed = parse_user_message(prompt)
            response = execute_intent(
                parsed, models,
                encode_geohash_fn=encode_geohash,
                generate_recommendations_fn=generate_recommendations
            )

        # Display assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # Sidebar helper for chat
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Chat Quick Actions:**")
    if st.sidebar.button("🔄 Clear Chat History"):
        st.session_state.chat_messages = [st.session_state.chat_messages[0]]
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 6 — MODEL MONITORING
# ═══════════════════════════════════════════════════════════════
elif page == "📡 Model Monitoring":
    st.title("📡 Model Monitoring & Drift Detection")
    st.markdown(
        "Track prediction quality, detect data drift, and monitor model health over time."
    )

    # Load prediction logs
    pred_logs = prediction_logger.get_all_predictions()

    if pred_logs.empty:
        st.info(
            "📭 No prediction logs yet. Use the **🔮 Event Impact Predictor** or "
            "**💬 ASTRAM Assistant** to generate predictions, and they'll appear here."
        )
    else:
        # Overview metrics
        st.subheader("📊 Prediction Volume")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Predictions", f"{len(pred_logs):,}")

        # Recent predictions (last 24h)
        recent = prediction_logger.get_recent_predictions(hours=24)
        mc2.metric("Last 24 Hours", f"{len(recent):,}")

        # Average priority risk
        prio_vals = pd.to_numeric(pred_logs['priority_prob'], errors='coerce').dropna()
        if len(prio_vals) > 0:
            mc3.metric("Avg Priority Risk", f"{prio_vals.mean()*100:.1f}%")

        # Average closure risk
        close_vals = pd.to_numeric(pred_logs['closure_prob'], errors='coerce').dropna()
        if len(close_vals) > 0:
            mc4.metric("Avg Closure Risk", f"{close_vals.mean()*100:.1f}%")

        # Prediction distribution over time
        st.subheader("📈 Prediction Distributions Over Time")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            if len(prio_vals) > 0:
                fig_prio_hist = px.histogram(
                    x=prio_vals * 100,
                    nbins=20,
                    title="Priority Risk Distribution (%)",
                    labels={'x': 'Priority Probability (%)'},
                    color_discrete_sequence=['#ef4444']
                )
                fig_prio_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'family': 'Outfit'},
                    showlegend=False
                )
                st.plotly_chart(fig_prio_hist, use_container_width=True)

        with col_m2:
            if len(close_vals) > 0:
                fig_close_hist = px.histogram(
                    x=close_vals * 100,
                    nbins=20,
                    title="Closure Risk Distribution (%)",
                    labels={'x': 'Closure Probability (%)'},
                    color_discrete_sequence=['#3b82f6']
                )
                fig_close_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white', 'family': 'Outfit'},
                    showlegend=False
                )
                st.plotly_chart(fig_close_hist, use_container_width=True)

        # Drift detection
        st.subheader("🔍 Drift Detection (PSI — Population Stability Index)")
        st.markdown(
            "PSI measures how much prediction distributions have shifted. "
            "Values **< 0.10** are stable, **0.10–0.25** indicate moderate drift, "
            "**> 0.25** suggests significant drift requiring retraining."
        )

        if len(pred_logs) >= 40:
            detector = DriftDetector()
            drift_results = detector.check_prediction_drift(pred_logs)

            drift_rows = []
            for col, info in drift_results.items():
                if isinstance(info, dict) and 'psi' in info:
                    drift_rows.append({
                        'Metric': col.replace('_', ' ').title(),
                        'PSI': info.get('psi', 'N/A'),
                        'Status': info.get('status', 'Unknown')
                    })

            if drift_rows:
                st.dataframe(
                    pd.DataFrame(drift_rows),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Drift analysis returned no results.")
        else:
            st.warning(
                f"Need at least 40 predictions for drift detection. "
                f"Current count: {len(pred_logs)}."
            )

        # Recent prediction log table
        st.subheader("📋 Recent Prediction Log")
        display_cols = ['timestamp', 'event_cause', 'zone', 'priority_prob',
                        'closure_prob', 'duration_est_min']
        available_cols = [c for c in display_cols if c in pred_logs.columns]
        st.dataframe(
            pred_logs[available_cols].tail(20).sort_values('timestamp', ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Online learning status
        st.subheader("🔄 Online Learning Status")
        retrain_log_path = 'models/retrain_log.json'
        if os.path.exists(retrain_log_path):
            import json
            with open(retrain_log_path, 'r') as f:
                retrain_logs = json.load(f)
            if retrain_logs:
                last = retrain_logs[-1]
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Last Retrain", last.get('timestamp', 'N/A')[:19])
                rc2.metric("Status", last.get('status', 'N/A'))
                rc3.metric("Data Rows", f"{last.get('data_rows', 'N/A'):,}"
                           if isinstance(last.get('data_rows'), int) else 'N/A')
        else:
            st.info(
                "No retraining has been performed yet. Run the incremental "
                "retraining script:\n\n"
                "```bash\npython online_update.py --retrain --window-months 6\n```"
            )

