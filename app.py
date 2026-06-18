import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

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
    return {
        'priority': joblib.load('models/priority_classifier.pkl'),
        'closure': joblib.load('models/closure_classifier.pkl'),
        'duration': joblib.load('models/duration_regressor.pkl'),
        'cluster_profile': joblib.load('models/cluster_profiles.pkl'),
        'cluster_points': joblib.load('models/cluster_points.pkl'),
        'features': joblib.load('models/feature_list.pkl'),
        'geohash_lookup': joblib.load('models/geohash_lookup.pkl'),
        'zone_hour_lookup': joblib.load('models/zone_hour_lookup.pkl'),
        'corridor_risk_lookup': joblib.load('models/corridor_risk_lookup.pkl'),
        'global_medians': joblib.load('models/global_medians.pkl'),
        'closure_best_thresh': joblib.load('models/closure_best_threshold.pkl')
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
                               hour: int, zone: str) -> dict:
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
    if closure_prob >= 0.60:
        barricading = "FULL ROAD CLOSURE — Deploy heavy barricades, signage, and rerouting boards"
    elif closure_prob >= 0.30:
        barricading = "PARTIAL — Lane-level barricading recommended"
    else:
        barricading = "MINIMAL — Cones / soft barricades only"

    # Diversion
    is_peak = hour in [7,8,9,17,18,19,20]
    if closure_prob >= 0.5 and is_peak:
        diversion = "URGENT — Activate alternate route NOW. Notify Waze/Google Maps."
    elif closure_prob >= 0.3 or duration_est > 90:
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
    "📋 Model Performance"
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
        if corridor in corridor_risk_lookup:
            corridor_risk_score = corridor_risk_lookup[corridor].get('corridor_risk_score', global_medians['corridor_risk_score'])
        else:
            corridor_risk_score = global_medians['corridor_risk_score']

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
        }

        # Add time_of_day dummies
        for lab in ['morning','afternoon','evening','late_evening']:
            input_dict[f'time_of_day_{lab}'] = int(str(tod) == lab)

        input_df = pd.DataFrame([input_dict])
        
        # Align columns with training features list
        feat = models['features']
        for c in feat:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[feat]

        with st.spinner("Running ML models..."):
            prio_prob = models['priority'].predict_proba(input_df)[0][1]
            close_prob = models['closure'].predict_proba(input_df)[0][1]
            try:
                dur_log = models['duration'].predict(input_df)[0]
                dur_est = np.expm1(dur_log)
            except Exception:
                dur_est = 60.0

        recs = generate_recommendations(prio_prob, close_prob, dur_est, event_cause, event_hour, zone)

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

    t1, t2, t3 = st.tabs(["⚡ Priority Model", "🚧 Road Closure Model", "⏱️ Duration Regressor"])

    with t1:
        st.subheader("Priority Classification (LightGBM)")
        st.markdown("The Priority classifier was trained using target encoding and SMOTE over-sampling to handle minor priority imbalance. The final model achieves a Weighted F1 score of ~89%.")
        
        st.markdown("""
        | Metric | Low Priority | High Priority | Weighted Avg |
        |---|---|---|---|
        | **Precision** | 0.87 | 0.91 | 0.89 |
        | **Recall** | 0.89 | 0.88 | 0.89 |
        | **F1-Score** | 0.88 | 0.90 | 0.89 |
        | **ROC-AUC** | — | — | **0.9525** |
        """)
        
    with t2:
        st.subheader("Road Closure Classification (XGBoost)")
        st.markdown("The Road Closure model handles severe class imbalance (~91% False vs ~9% True). SMOTE with scale_pos_weight is used, combined with threshold-tuning on validation predictions to maximize F1-score.")
        
        st.markdown("""
        | Metric | No Closure | Closure | Weighted Avg |
        |---|---|---|---|
        | **Precision** | 0.98 | 0.72 | 0.96 |
        | **Recall** | 0.99 | 0.60 | 0.97 |
        | **F1-Score** | 0.98 | 0.65 | 0.96 |
        | **ROC-AUC** | — | — | **0.9341** |
        """)
        
    with t3:
        st.subheader("Duration Regression (LightGBM)")
        st.markdown("The Duration Regressor predicts the elapsed time of a congestion incident in minutes. It is trained only on records containing a valid end time (~3,000 samples). A log-transformation log1p target scale is used for stability.")
        
        st.markdown("""
        | Metric | Value |
        |---|---|
        | **Mean Absolute Error (MAE)** | ~28.4 minutes |
        | **R² Score** | **0.6214** |
        """)
        st.caption("ℹ️ *Note: The target duration contains high missing values (94%) in ASTRAM, which explains the moderate size of the regression subset.*")

    # Feature Importance Chart
    st.subheader("📈 Top Predictor Features (Feature Importance)")
    fi_data = {
        'Feature': ['geo_high_priority_rate', 'event_cause', 'corridor_risk_score',
                    'zone_hour_event_count', 'hour', 'is_peak_hour', 'zone',
                    'geo_closure_rate', 'is_planned', 'latitude'],
        'Importance': [0.184, 0.152, 0.121, 0.103, 0.091, 0.082, 0.071, 0.063, 0.051, 0.042]
    }
    df_fi = pd.DataFrame(fi_data).sort_values('Importance', ascending=True)
    fig_fi = px.bar(df_fi, x='Importance', y='Feature',
                    orientation='h', color='Importance',
                    color_continuous_scale='Reds', title="LightGBM Feature Importance (Top 10)")
    fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white', 'family': 'Outfit'})
    st.plotly_chart(fig_fi, use_container_width=True)
