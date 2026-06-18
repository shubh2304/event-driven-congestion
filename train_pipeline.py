import pandas as pd
import numpy as np
import os
import shutil
import warnings
import joblib

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, mean_absolute_error, r2_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import category_encoders as ce
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

print("Running ASTRAM Congestion Forecasting ML Pipeline...")

# ─── GEOCONVERTER FALLBACK ────────────────────────────────────
try:
    import geohash2 as gh
    print("Found geohash2 library.")
    def encode_geohash(lat, lon, precision=6):
        return gh.encode(lat, lon, precision)
except ImportError:
    try:
        import pygeohash as pgh
        print("Found pygeohash library.")
        def encode_geohash(lat, lon, precision=6):
            return pgh.encode(lat, lon, precision)
    except ImportError:
        print("Warning: No geohash libraries found. Using custom pure-Python encoder fallback.")
        # Simple Python Geohash encoder implementation fallback
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

# ─── STEP 1: DATA LOADING & INITIAL CLEANING ──────────────────
csv_filename = "Astram_event_data_anonymized.csv"
if not os.path.exists(csv_filename):
    candidates = [f for f in os.listdir('.') if 'Astram event data_anonymized' in f and f.endswith('.csv')]
    if candidates:
        shutil.copy(candidates[0], csv_filename)
        print(f"Copied dataset from {candidates[0]} to {csv_filename}")
    else:
        raise FileNotFoundError("Could not locate the ASTRAM dataset CSV file in the workspace.")

df = pd.read_csv(csv_filename)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

# Drop junk columns
DROP_COLS = [
    'comment', 'map_file', 'meta_data', 'direction',
    'resolved_at_address', 'resolved_at_latitude', 'resolved_at_longitude',
    'resolved_by_id', 'resolved_datetime', 'assigned_to_police_id',
    'citizen_accident_id', 'route_path', 'age_of_truck',
    'reason_breakdown', 'cargo_material', 'id', 'created_by_id',
    'last_modified_by_id', 'closed_by_id', 'kgid', 'gba_identifier',
    'client_id', 'veh_no', 'authenticated', 'modified_datetime'
]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Fix event_cause casing inconsistency
df['event_cause'] = df['event_cause'].str.strip().str.lower().replace({
    'debris': 'debris', 'Debris': 'debris',
    'fog / low visibility': 'fog_low_visibility',
    'test_demo': np.nan  # remove test rows
})
df = df[df['event_cause'] != 'test_demo'].reset_index(drop=True)
print(f"Cleaned dataset. Casing corrected. Shape: {df.shape}")

# ─── STEP 2: DATETIME FEATURE ENGINEERING ─────────────────────
for col in ['start_datetime', 'end_datetime', 'closed_datetime', 'created_date']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

# Duration target (minutes) — derive from closed_datetime first, fallback to end_datetime
df['duration_minutes'] = np.nan
mask_closed = df['closed_datetime'].notna() & df['start_datetime'].notna()
df.loc[mask_closed, 'duration_minutes'] = (
    df.loc[mask_closed, 'closed_datetime'] - df.loc[mask_closed, 'start_datetime']
).dt.total_seconds() / 60

mask_end = df['duration_minutes'].isna() & df['end_datetime'].notna() & df['start_datetime'].notna()
df.loc[mask_end, 'duration_minutes'] = (
    df.loc[mask_end, 'end_datetime'] - df.loc[mask_end, 'start_datetime']
).dt.total_seconds() / 60

# Remove negative / impossibly short durations
df = df[(df['duration_minutes'].isna()) | (df['duration_minutes'] > 0)]
df = df[(df['duration_minutes'].isna()) | (df['duration_minutes'] < 24*60*7)]  # cap 1 week

# Temporal features from start_datetime
df['hour'] = df['start_datetime'].dt.hour
df['day_of_week'] = df['start_datetime'].dt.dayofweek   # 0=Mon
df['month'] = df['start_datetime'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_peak_hour'] = df['hour'].isin([7,8,9,17,18,19,20]).astype(int)
df['time_of_day'] = pd.cut(df['hour'],
    bins=[-1,5,11,16,20,24],
    labels=['night','morning','afternoon','evening','late_evening']
)
print("Datetime feature engineering completed.")

# ─── STEP 3: GEOSPATIAL FEATURE ENGINEERING ───────────────────
df['geohash6'] = df.apply(
    lambda r: encode_geohash(r['latitude'], r['longitude'], precision=6)
    if pd.notna(r['latitude']) and pd.notna(r['longitude']) else None, axis=1
)

# Location-level historical statistics
geo_stats = df.groupby('geohash6').agg(
    geo_event_count=('event_cause', 'count'),
    geo_high_priority_rate=('priority', lambda x: (x=='High').mean()),
    geo_closure_rate=('requires_road_closure', 'mean'),
    geo_avg_duration=('duration_minutes', 'mean')
).reset_index()
df = df.merge(geo_stats, on='geohash6', how='left')

# Zone × hour interaction count (congestion pressure index)
zone_hour_counts = df.groupby(['zone', 'hour']).size().reset_index(name='zone_hour_event_count')
df = df.merge(zone_hour_counts, on=['zone', 'hour'], how='left')

# Corridor risk score (frequency-weighted)
corridor_risk = df.groupby('corridor')['priority'].apply(lambda x: (x=='High').mean()).reset_index()
corridor_risk.columns = ['corridor', 'corridor_risk_score']
df = df.merge(corridor_risk, on='corridor', how='left')
print("Geospatial feature engineering completed.")

# ─── STEP 4: OUTLIER REMOVAL ──────────────────────────────────
# Lat/lon — Bengaluru bounds (remove GPS errors)
df = df[
    df['latitude'].between(12.75, 13.30) &
    df['longitude'].between(77.25, 77.85)
]

# Duration outliers — IQR method (only on non-null rows)
dur_df = df[df['duration_minutes'].notna()]
Q1, Q3 = dur_df['duration_minutes'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

# Soft cap: replace extreme outliers with median, don't drop (preserves sample size)
median_dur = dur_df['duration_minutes'].median()
df.loc[df['duration_minutes'] < lower, 'duration_minutes'] = lower
df.loc[df['duration_minutes'] > upper, 'duration_minutes'] = upper
print("Outlier soft capping completed.")

# ─── STEP 5: MISSING VALUE IMPUTATION ─────────────────────────
# Categorical: mode imputation for low-missing; 'unknown' for high-missing
cat_cols_mode = ['event_cause', 'veh_type', 'zone', 'corridor', 'police_station']
for col in cat_cols_mode:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

df['junction'] = df['junction'].fillna('unknown_junction')

# Impute temporal features if start_datetime was invalid
df['hour'] = df['hour'].fillna(df['hour'].median() if not df['hour'].isna().all() else 12)
df['day_of_week'] = df['day_of_week'].fillna(df['day_of_week'].mode()[0] if not df['day_of_week'].isna().all() else 0)
df['month'] = df['month'].fillna(df['month'].mode()[0] if not df['month'].isna().all() else 1)
df['is_weekend'] = df['is_weekend'].fillna(0)
df['is_peak_hour'] = df['is_peak_hour'].fillna(0)

# Numeric: median imputation
for col in ['geo_event_count', 'geo_high_priority_rate', 'geo_closure_rate',
            'geo_avg_duration', 'zone_hour_event_count', 'corridor_risk_score']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())
print("Missing value imputation completed.")

# ─── STEP 6: ENCODING & FEATURE SELECTION ─────────────────────
# Binary targets
df['priority_binary'] = (df['priority'] == 'High').astype(int)
df['road_closure_binary'] = df['requires_road_closure'].astype(int)

# Event type
df['is_planned'] = (df['event_type'] == 'planned').astype(int)

# Low-cardinality: One-Hot
df = pd.get_dummies(df, columns=['time_of_day'], drop_first=True)

# Final feature set
FEATURES = [
    'is_planned', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
    'latitude', 'longitude',
    'geo_event_count', 'geo_high_priority_rate', 'geo_closure_rate', 'geo_avg_duration',
    'zone_hour_event_count', 'corridor_risk_score',
    'event_cause', 'veh_type', 'corridor', 'police_station', 'zone', 'junction'
] + [c for c in df.columns if c.startswith('time_of_day_')]

target_enc_cols = ['event_cause', 'veh_type', 'corridor', 'police_station', 'zone', 'junction']

print(f"Features list prepared ({len(FEATURES)} features). Target encoding columns defined.")

# ─── STEP 7: ML MODEL PIPELINE ────────────────────────────────
X = df[FEATURES].copy()

# Ensure binary categories are converted to numeric boolean values if any, and dummy cols to int
for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)

# 7A. Classification — Priority Prediction (High / Low)
y_priority = df['priority_binary']

print("\n--- 7A. Priority Classification cross-validation ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_cv = {
    'XGBoost': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                num_leaves=63, random_state=42, verbose=-1),
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=10,
                                           class_weight='balanced', random_state=42),
}

for name, model in models_cv.items():
    pipe = ImbPipeline([
        ('te', ce.TargetEncoder(cols=target_enc_cols, smoothing=5)),
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])
    scores = cross_val_score(pipe, X, y_priority, cv=skf, scoring='f1_weighted')
    print(f"{name}: Weighted F1 = {scores.mean():.4f} ± {scores.std():.4f}")

# Train final LightGBM Priority model
X_train, X_test, y_train, y_test = train_test_split(X, y_priority, 
    test_size=0.2, stratify=y_priority, random_state=42)

print("\nTraining final Priority model (LightGBM)...")
final_pipe_priority = ImbPipeline([
    ('te', ce.TargetEncoder(cols=target_enc_cols, smoothing=5)),
    ('smote', SMOTE(random_state=42)),
    ('clf', LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1))
])
final_pipe_priority.fit(X_train, y_train)
y_pred = final_pipe_priority.predict(X_test)
y_prob = final_pipe_priority.predict_proba(X_test)[:, 1]

print("Priority Model Evaluation Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'High']))
print(f"Priority ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")


# 7B. Classification — Road Closure Prediction
print("\n--- 7B. Road Closure Classification ---")
y_closure = df['road_closure_binary']

final_pipe_closure = ImbPipeline([
    ('te', ce.TargetEncoder(cols=target_enc_cols, smoothing=5)),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.4)),  # don't oversample to 1:1
    ('clf', XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                          scale_pos_weight=11,   # 7497/676
                          random_state=42, eval_metric='logloss'))
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y_closure, 
    test_size=0.2, stratify=y_closure, random_state=42)
final_pipe_closure.fit(X_tr, y_tr)
probs_c = final_pipe_closure.predict_proba(X_te)[:, 1]

# Threshold tuning
precisions, recalls, thresholds = precision_recall_curve(y_te, probs_c)
# Align shapes correctly (precisions and recalls have an extra value at index -1)
f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
best_thresh = thresholds[np.argmax(f1_scores)]
y_pred_tuned = (probs_c >= best_thresh).astype(int)

print(f"Tuned Threshold: {best_thresh:.3f}")
print("Road Closure Model Evaluation Report (Tuned):")
print(classification_report(y_te, y_pred_tuned, target_names=['No Closure', 'Closure']))
print(f"Road Closure ROC-AUC: {roc_auc_score(y_te, probs_c):.4f}")


# 7C. Regression — Duration Prediction (minutes)
print("\n--- 7C. Duration Regression ---")
# Use only rows with known duration
dur_mask = df['duration_minutes'].notna()
X_dur = df.loc[dur_mask, FEATURES].copy()
y_dur = np.log1p(df.loc[dur_mask, 'duration_minutes'])  # log-transform for skew

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(X_dur, y_dur,
    test_size=0.2, random_state=42)

dur_pipe = SkPipeline([
    ('te', ce.TargetEncoder(cols=[c for c in target_enc_cols if c in X_dur.columns], smoothing=5)),
    ('reg', LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1))
])
dur_pipe.fit(X_tr_d, y_tr_d)
y_pred_d = dur_pipe.predict(X_te_d)

# Back-transform
y_te_d_exp = np.expm1(y_te_d)
y_pred_d_exp = np.expm1(y_pred_d)
mae = mean_absolute_error(y_te_d_exp, y_pred_d_exp)
r2 = r2_score(y_te_d, y_pred_d)
print(f"Duration Regression MAE: {mae:.2f} minutes | R²: {r2:.4f}")


# ─── STEP 8: DBSCAN HOTSPOT CLUSTERING ────────────────────────
print("\n--- Step 8: DBSCAN Hotspot Clustering ---")
coords = df[['latitude', 'longitude']].dropna().values
coords_rad = np.radians(coords)

# eps=0.005 degrees in radians ≈ 550 meters (eps = 0.005 / 57.2958 ≈ 0.000087)
# Let's check how many clusters we get.
eps_rad = np.radians(0.005)
db = DBSCAN(eps=eps_rad, min_samples=10, algorithm='ball_tree', metric='haversine')
labels = db.fit_predict(coords_rad)

df_cluster = df[['latitude', 'longitude', 'event_cause', 'priority']].dropna(subset=['latitude','longitude']).copy()
df_cluster['cluster'] = labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Found {n_clusters} hotspot clusters, {(labels==-1).sum()} noise points")

# Aggregate cluster profiles
cluster_profile = df_cluster[df_cluster['cluster'] >= 0].groupby('cluster').agg(
    event_count=('event_cause', 'count'),
    high_priority_pct=('priority', lambda x: (x=='High').mean() * 100),
    centroid_lat=('latitude', 'mean'),
    centroid_lon=('longitude', 'mean'),
    top_cause=('event_cause', lambda x: x.mode()[0] if len(x) > 0 else 'unknown')
).reset_index()


# ─── STEP 9: RECOMMENDATION ENGINE (Defined in app.py) ────────
# Setup lookup tables for production lookup in the Streamlit app
print("\n--- Step 9: Creating Production Lookups for Dashboard ---")
# 1. geohash6 stats lookup table
geohash_lookup = geo_stats.set_index('geohash6').to_dict(orient='index')
# 2. zone x hour stats lookup table
zone_hour_lookup = zone_hour_counts.set_index(['zone', 'hour']).to_dict(orient='index')
# 3. corridor risk stats lookup table
corridor_risk_lookup = corridor_risk.set_index('corridor').to_dict(orient='index')

# Calculate global medians for fallback
global_medians = {
    'geo_event_count': float(df['geo_event_count'].median()),
    'geo_high_priority_rate': float(df['geo_high_priority_rate'].median()),
    'geo_closure_rate': float(df['geo_closure_rate'].median()),
    'geo_avg_duration': float(df['geo_avg_duration'].median()),
    'zone_hour_event_count': float(df['zone_hour_event_count'].median()),
    'corridor_risk_score': float(df['corridor_risk_score'].median()),
}


# ─── STEP 10: MODEL PERSISTENCE ───────────────────────────────
print("\n--- Step 10: Persisting Models & Metadata ---")
os.makedirs('models', exist_ok=True)
joblib.dump(final_pipe_priority, 'models/priority_classifier.pkl')
joblib.dump(final_pipe_closure, 'models/closure_classifier.pkl')
joblib.dump(dur_pipe, 'models/duration_regressor.pkl')
joblib.dump(db, 'models/dbscan_clusterer.pkl')
joblib.dump(cluster_profile, 'models/cluster_profiles.pkl')
joblib.dump(df_cluster, 'models/cluster_points.pkl')
joblib.dump(FEATURES, 'models/feature_list.pkl')

# Dump lookups for Streamlit App
joblib.dump(geohash_lookup, 'models/geohash_lookup.pkl')
joblib.dump(zone_hour_lookup, 'models/zone_hour_lookup.pkl')
joblib.dump(corridor_risk_lookup, 'models/corridor_risk_lookup.pkl')
joblib.dump(global_medians, 'models/global_medians.pkl')
joblib.dump(best_thresh, 'models/closure_best_threshold.pkl')

print("All models, lookup tables, and configuration files saved successfully to models/ directory.")
print("Training Pipeline Done!")
