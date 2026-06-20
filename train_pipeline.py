import pandas as pd
import numpy as np
import os
import shutil
import warnings
import joblib

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, mean_absolute_error, r2_score, precision_recall_curve
from sklearn.base import BaseEstimator, TransformerMixin
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

# ─── PREPROCESSING TRANSFORMERS ─────────────────────────────────────────────
# Sklearn-compatible transformers fitted exclusively on df_train (after the
# primary split in Step 6.5).  Their fitted values are bundled into
# models/preprocessor.pkl so app.py applies the EXACT same transforms at
# inference without any re-computation or hardcoded fallback logic.

class IQRDurationCapper(BaseEstimator, TransformerMixin):
    """Soft-caps duration_minutes using IQR bounds derived from training data.

    fit():      computes Q1, Q3 from df_train[col] (non-null only) and stores
                lower_ = Q1 - factor*IQR  and  upper_ = Q3 + factor*IQR.
    transform(): clips values to [lower_, upper_] — no rows are dropped.
    """
    def __init__(self, col='duration_minutes', factor=1.5):
        self.col    = col
        self.factor = factor

    def fit(self, df, y=None):
        dur  = df[self.col].dropna()
        q1, q3       = float(dur.quantile(0.25)), float(dur.quantile(0.75))
        iqr          = q3 - q1
        self.lower_  = q1 - self.factor * iqr
        self.upper_  = q3 + self.factor * iqr
        return self

    def transform(self, df, y=None):
        df   = df.copy()
        mask = df[self.col].notna()
        df.loc[mask, self.col] = df.loc[mask, self.col].clip(self.lower_, self.upper_)
        return df


class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Fills NaN in categorical columns using mode values from training data.

    mode_cols:   list of column names whose mode is computed at fit() time.
    fixed_fills: {col: constant} applied regardless of training distribution
                 (e.g. junction → 'unknown_junction').
    """
    def __init__(self, mode_cols=None, fixed_fills=None):
        self.mode_cols   = mode_cols   or []
        self.fixed_fills = fixed_fills or {}

    def fit(self, df, y=None):
        self.mode_values_ = {}
        for col in self.mode_cols:
            if col in df.columns and not df[col].isna().all():
                self.mode_values_[col] = df[col].mode()[0]
        return self

    def transform(self, df, y=None):
        df = df.copy()
        for col, val in self.mode_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        for col, val in self.fixed_fills.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df


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

# ─── STEP 3: GEOHASH ENCODING ONLY ───────────────────────────
# NOTE: Geo aggregations (geo_stats, zone_hour_counts, corridor_risk) are
# intentionally deferred to Step 6.6, which runs AFTER the primary
# train/test split. Computing them here — on the full df — would leak
# test-set labels into training features (e.g. geo_high_priority_rate would
# include each row's own priority label in its own geohash aggregate).
df['geohash6'] = df.apply(
    lambda r: encode_geohash(r['latitude'], r['longitude'], precision=6)
    if pd.notna(r['latitude']) and pd.notna(r['longitude']) else None, axis=1
)
print("Geohash encoding completed. Geo aggregations deferred until after train/test split.")

# ─── STEP 4: LAT/LON FILTERING ──────────────────────────────────────────────
# Remove GPS errors that fall outside Bengaluru city bounds.
# IQR soft-capping of duration_minutes is intentionally deferred to Step 6.7
# where IQRDurationCapper is fitted on df_train only and saved to
# models/preprocessor.pkl — ensuring (a) no test-set leakage in the bounds
# and (b) consistent capping at inference time.
df = df[
    df['latitude'].between(12.75, 13.30) &
    df['longitude'].between(77.25, 77.85)
]
print("Lat/lon bounds filtering completed.")

# ─── STEP 5: PRE-SPLIT MINIMAL IMPUTATION ───────────────────────────────────
# Only deterministic, non-learnable fills are applied here (before the split).
# Mode/median-based categorical imputation is deferred to Step 6.7 where
# CategoricalImputer is fitted on df_train only — no full-dataset statistics
# contaminate the imputed values used during training.
#
# junction: always fill with a known constant (no meaningful mode exists)
df['junction'] = df['junction'].fillna('unknown_junction')
# Temporal features: safe neutral defaults for the rare edge case where
# start_datetime is entirely missing (affects ~0 rows in practice)
df['hour']        = df['hour'].fillna(12)       # noon — neutral time-of-day default
df['day_of_week'] = df['day_of_week'].fillna(0) # Monday — neutral weekday default
df['month']       = df['month'].fillna(1)        # January — neutral month default
df['is_weekend']  = df['is_weekend'].fillna(0)
df['is_peak_hour'] = df['is_peak_hour'].fillna(0)
# Geo-aggregate columns are not yet present — NaN fill handled in Step 6.7.
print("Pre-split imputation completed.")

# ─── STEP 6: BASE ENCODING ────────────────────────────────────
# Binary targets
df['priority_binary'] = (df['priority'] == 'High').astype(int)
df['road_closure_binary'] = df['requires_road_closure'].astype(int)

# Event type
df['is_planned'] = (df['event_type'] == 'planned').astype(int)

# Low-cardinality: One-Hot (applied to full df so all categories are seen)
df = pd.get_dummies(df, columns=['time_of_day'], drop_first=True)

target_enc_cols = ['event_cause', 'veh_type', 'corridor', 'police_station', 'zone', 'junction']
print("Base encoding completed.")

# ─── STEP 6.5: PRIMARY TRAIN / TEST SPLIT ─────────────────────
# This split MUST happen before any target-aware geo aggregations.
# All three model evaluations (7A, 7B, 7C) share this same split for
# consistent and fair comparison.
print("\n--- Step 6.5: Primary Train/Test Split (pre-aggregation) ---")
df_train, df_test = train_test_split(
    df, test_size=0.2, stratify=df['priority_binary'], random_state=42
)
df_train = df_train.copy().reset_index(drop=True)
df_test  = df_test.copy().reset_index(drop=True)
print(f"Train: {len(df_train)} rows | Test: {len(df_test)} rows")

# ─── STEP 6.6: GEO AGGREGATIONS ON TRAINING DATA ONLY ─────────
# All target-aware aggregations are computed exclusively from df_train.
# df_test rows look up their values from these training-derived tables;
# unseen geohashes / corridors fall back to training medians.
print("\n--- Step 6.6: Geospatial Aggregations (train-only) ---")

geo_stats = df_train.groupby('geohash6').agg(
    geo_event_count=('event_cause', 'count'),
    geo_high_priority_rate=('priority', lambda x: (x == 'High').mean()),
    geo_closure_rate=('requires_road_closure', 'mean'),
    geo_avg_duration=('duration_minutes', 'mean')
).reset_index()

zone_hour_counts = df_train.groupby(['zone', 'hour']).size().reset_index(name='zone_hour_event_count')

corridor_risk = df_train.groupby('corridor')['priority'].apply(
    lambda x: (x == 'High').mean()
).reset_index()
corridor_risk.columns = ['corridor', 'corridor_risk_score']
corridor_risk['corridor'] = corridor_risk['corridor'].str.strip().str.lower()
df_train['corridor'] = df_train['corridor'].str.strip().str.lower()
df_test['corridor']  = df_test['corridor'].str.strip().str.lower()

# Merge onto train (all match) and test (unseen geohashes → NaN → filled below)
df_train = df_train.merge(geo_stats,        on='geohash6',         how='left')
df_test  = df_test.merge(geo_stats,         on='geohash6',         how='left')
df_train = df_train.merge(zone_hour_counts, on=['zone', 'hour'],   how='left')
df_test  = df_test.merge(zone_hour_counts,  on=['zone', 'hour'],   how='left')
df_train = df_train.merge(corridor_risk,    on='corridor',         how='left')
df_test  = df_test.merge(corridor_risk,     on='corridor',         how='left')

# RC-1: Per-cause historical closure rate (train-only, leakage-free)
# Encodes the empirical probability that each event cause leads to a road closure.
# e.g. accidents close roads ~22% of the time, potholes almost never do.
# This is strictly train-derived — same safe pattern as geo_closure_rate.
cause_closure = df_train.groupby('event_cause')['road_closure_binary'].mean().reset_index()
cause_closure.columns = ['event_cause', 'cause_closure_rate']
df_train = df_train.merge(cause_closure, on='event_cause', how='left')
df_test  = df_test.merge(cause_closure,  on='event_cause', how='left')

# DR-1: Per-cause historical average duration (train-only, leakage-free)
cause_duration = df_train.groupby('event_cause')['duration_minutes'].mean().reset_index()
cause_duration.columns = ['event_cause', 'cause_avg_duration']
df_train = df_train.merge(cause_duration, on='event_cause', how='left')
df_test  = df_test.merge(cause_duration,  on='event_cause', how='left')

# Fallback medians computed from TRAIN ONLY — saved for app.py inference
geo_train_medians = {
    'geo_event_count':        float(df_train['geo_event_count'].median()),
    'geo_high_priority_rate': float(df_train['geo_high_priority_rate'].median()),
    'geo_closure_rate':       float(df_train['geo_closure_rate'].median()),
    'geo_avg_duration':       float(df_train['geo_avg_duration'].median()),
    'zone_hour_event_count':  float(df_train['zone_hour_event_count'].median()),
    'corridor_risk_score':    float(df_train['corridor_risk_score'].median()),
    'cause_closure_rate':     float(df_train['cause_closure_rate'].median()),
    'cause_avg_duration':     float(df_train['cause_avg_duration'].median()),
}
for col, med_val in geo_train_medians.items():
    df_train[col] = df_train[col].fillna(med_val)
    df_test[col]  = df_test[col].fillna(med_val)
print("Geospatial aggregations complete. Train medians saved for inference fallback.")

# ─── STEP 6.7: FIT PREPROCESSING TRANSFORMERS ON TRAINING DATA ───────────────
# Replaces the old bare-Pandas IQR capping (was Step 4) and mode imputation
# (was Step 5) that were both computed on the full df before the split.
# These sklearn transformers are fitted on df_train only → no test-set
# contamination of capping bounds or mode-imputation values.
print("\n--- Step 6.7: Fitting Preprocessing Transformers (train-only) ---")

iqr_capper = IQRDurationCapper(col='duration_minutes', factor=1.5)
iqr_capper.fit(df_train)
print(f"IQRDurationCapper: lower={iqr_capper.lower_:.1f} min | upper={iqr_capper.upper_:.1f} min")

cat_imputer = CategoricalImputer(
    mode_cols=['event_cause', 'veh_type', 'zone', 'corridor', 'police_station'],
    fixed_fills={'junction': 'unknown_junction'}
)
cat_imputer.fit(df_train)
print(f"CategoricalImputer modes (train-only): {cat_imputer.mode_values_}")

df_train = iqr_capper.transform(df_train)
df_test  = iqr_capper.transform(df_test)    # same training bounds applied to test — no leakage
df_train = cat_imputer.transform(df_train)
df_test  = cat_imputer.transform(df_test)   # same training modes applied to test — no leakage
print("Preprocessing transformers applied to train and test splits.")

# ─── FEATURE LISTS ─────────────────────────────────────────────
# Exclusions from FEATURES_PRIORITY:
#   geo_high_priority_rate  — smoothed copy of the priority target label
#   corridor                — data investigation showed this IS the priority rule:
#                             Non-corridor → 100% Low (3,120/3,120)
#                             Named corridor → 99.6% High (5,029/5,048)
#   corridor_risk_score     — derived from corridor, encodes the same rule as a float
# Both corridor features are kept in FEATURES_ALL where they are legitimate
# predictors for the closure and duration models (different targets).
TIME_DUMMIES = sorted([c for c in df_train.columns if c.startswith('time_of_day_')])

FEATURES_PRIORITY = [
    'is_planned', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
    'latitude', 'longitude',
    'geo_event_count', 'geo_closure_rate', 'geo_avg_duration',
    'zone_hour_event_count',
    # corridor and corridor_risk_score excluded — they are the labeling rule, not a feature
    # geo_high_priority_rate excluded — smoothed copy of the priority target
    'event_cause', 'veh_type', 'police_station', 'zone', 'junction'
] + TIME_DUMMIES

# TargetEncoder for priority model must only reference columns that are IN FEATURES_PRIORITY
# (corridor is excluded, so it must be removed from the encoder's cols list too)
target_enc_cols_priority = [c for c in target_enc_cols if c in FEATURES_PRIORITY]

FEATURES_ALL = [
    'is_planned', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
    'latitude', 'longitude',
    'geo_event_count', 'geo_high_priority_rate', 'geo_closure_rate', 'geo_avg_duration',
    'zone_hour_event_count', 'corridor_risk_score',
    'event_cause', 'veh_type', 'corridor', 'police_station', 'zone', 'junction',
    'cause_avg_duration',  # DR-1
] + TIME_DUMMIES

# FEATURES_CLOSURE: superset of FEATURES_ALL with closure-specific engineered features.
# Kept separate from FEATURES_ALL so Duration model is not polluted by closure-target
# leaky-adjacent signals (cause_closure_rate encodes the closure target).
#
# RC-2: is_high_risk_cause  — binary flag: causes that empirically drive road closures
# Derived programmatically from train-set only (causes with closure rate >= 0.3)
HIGH_RISK_CAUSES = set(cause_closure.loc[cause_closure['cause_closure_rate'] >= 0.3, 'event_cause'])
print(f"HIGH_RISK_CAUSES derived from train (rate >= 0.3): {HIGH_RISK_CAUSES}")
df_train['is_high_risk_cause'] = df_train['event_cause'].isin(HIGH_RISK_CAUSES).astype(int)
df_test['is_high_risk_cause']  = df_test['event_cause'].isin(HIGH_RISK_CAUSES).astype(int)

FEATURES_CLOSURE = FEATURES_ALL + [
    'cause_closure_rate',   # RC-1
    'is_high_risk_cause',   # RC-2
]

print(f"FEATURES_PRIORITY: {len(FEATURES_PRIORITY)} features "
      f"(corridor, corridor_risk_score, geo_high_priority_rate all excluded)")
print(f"FEATURES_ALL:      {len(FEATURES_ALL)} features (duration model — no closure-target signals)")
print(f"FEATURES_CLOSURE:  {len(FEATURES_CLOSURE)} features (closure model — includes RC-1, RC-2)")
print(f"target_enc_cols_priority: {target_enc_cols_priority}")

# ─── PREPROCESSING STATE ─────────────────────────────────────────────────────
# Bundles fitted transform values as plain Python types (no class objects) so
# models/preprocessor.pkl deserialises cleanly in app.py without needing the
# IQRDurationCapper / CategoricalImputer class definitions to be present.
preprocessor_state = {
    'time_dummies': TIME_DUMMIES,              # exact OHE column names: ['time_of_day_afternoon', ...]
    'iqr_lower':    float(iqr_capper.lower_),  # min duration allowed at inference
    'iqr_upper':    float(iqr_capper.upper_),  # max duration allowed at inference
    'mode_values':  dict(cat_imputer.mode_values_),   # categorical NaN fill values
}
print(f"Preprocessing state: time_dummies={TIME_DUMMIES}")
print(f"                     iqr=[{iqr_capper.lower_:.1f}, {iqr_capper.upper_:.1f}] min")
print(f"                     mode_values={cat_imputer.mode_values_}")

# ─── STEP 7: ML MODEL PIPELINES ───────────────────────────────
# df_train / df_test were defined in Step 6.5. All three models use this
# single, consistent split — no per-model train_test_split calls below.

def prep_X(df_subset, feature_list):
    """Extract and type-cast a feature matrix from a dataframe subset."""
    X = df_subset[feature_list].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X

# 7A. Classification — Priority Prediction (High / Low)
# Uses FEATURES_PRIORITY which excludes geo_high_priority_rate.
print("\n--- 7A. Priority Classification cross-validation ---")
X_train_p = prep_X(df_train, FEATURES_PRIORITY)
X_test_p  = prep_X(df_test,  FEATURES_PRIORITY)
y_train_p = df_train['priority_binary']
y_test_p  = df_test['priority_binary']

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

lgbm_f1_cv = 0.0
for name, model in models_cv.items():
    pipe = ImbPipeline([
        ('te', ce.TargetEncoder(cols=target_enc_cols_priority, smoothing=5)),
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])
    scores = cross_val_score(pipe, X_train_p, y_train_p, cv=skf, scoring='f1_weighted')
    if name == 'LightGBM':
        lgbm_f1_cv = scores.mean()
    print(f"{name}: Weighted F1 = {scores.mean():.4f} ± {scores.std():.4f}")

print("\nTraining final Priority model (LightGBM)...")
final_pipe_priority = ImbPipeline([
    ('te', ce.TargetEncoder(cols=target_enc_cols_priority, smoothing=5)),
    ('smote', SMOTE(random_state=42)),
    ('clf', LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1))
])
final_pipe_priority.fit(X_train_p, y_train_p)
y_pred = final_pipe_priority.predict(X_test_p)
y_prob = final_pipe_priority.predict_proba(X_test_p)[:, 1]

print("Priority Model Evaluation Report:")
print(classification_report(y_test_p, y_pred, target_names=['Low', 'High']))
print(f"Priority ROC-AUC: {roc_auc_score(y_test_p, y_prob):.4f}")

lgbm_model = final_pipe_priority.named_steps['clf']
te_step = final_pipe_priority.named_steps['te']
fi_scores = lgbm_model.feature_importances_
fi_df = pd.DataFrame({'feature': FEATURES_PRIORITY, 'importance': fi_scores})
fi_df = fi_df.sort_values('importance', ascending=False).head(10)
priority_feature_importance = fi_df.to_dict(orient='records')


# 7B. Classification — Road Closure Prediction
# RC-3 winner: XGBoost + SMOTE(0.6) — locked in.
# RC-4/RC-5: Compare LightGBM+is_unbalance and CatBoost+Balanced.
# RC-6: Add zone_peak_int and cause_risk_int interaction features.
# RC-7: Hyperparameter sweep on the overall winner.
# All thresholds tuned on val; all metrics reported on blind test set.
print("\n--- 7B. Road Closure Classification ---")

from sklearn.metrics import precision_score, recall_score, f1_score as sklearn_f1

# Carve out a 20% validation split from df_train (stratified on closure label).
df_tr_c, df_val_c = train_test_split(
    df_train, test_size=0.2, stratify=df_train['road_closure_binary'], random_state=42
)
y_tr  = df_tr_c['road_closure_binary']
y_val = df_val_c['road_closure_binary']
y_te  = df_test['road_closure_binary']

# Base feature matrices (RC-1+RC-2 features)
X_tr  = prep_X(df_tr_c,  FEATURES_CLOSURE)
X_val = prep_X(df_val_c, FEATURES_CLOSURE)
X_te  = prep_X(df_test,  FEATURES_CLOSURE)

# ── Final closure model — using the winning hyperparameters (RC-7: depth=8, lr=0.1) ─
_te_cols = [c for c in target_enc_cols if c in FEATURES_CLOSURE]
final_pipe_closure = ImbPipeline([
    ('te', ce.TargetEncoder(cols=_te_cols, smoothing=5)),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.6)),
    ('clf', XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.10,
                          random_state=42, eval_metric='logloss'))
])
final_pipe_closure.fit(X_tr, y_tr)

# ── Threshold tuning on VALIDATION set (never seen during fit) ────────────────
probs_val = final_pipe_closure.predict_proba(X_val)[:, 1]
precisions_v, recalls_v, thresholds_v = precision_recall_curve(y_val, probs_val)
f1_scores_v = (2 * precisions_v[:-1] * recalls_v[:-1]
               / (precisions_v[:-1] + recalls_v[:-1] + 1e-8))
best_thresh = float(thresholds_v[np.argmax(f1_scores_v)])

# ── Final evaluation on TEST set using the validation-derived threshold ────────
probs_c = final_pipe_closure.predict_proba(X_te)[:, 1]
y_pred_tuned = (probs_c >= best_thresh).astype(int)

print(f"Tuned Threshold (val-derived): {best_thresh:.3f}")
print("Road Closure Model Evaluation Report (Tuned):")
print(classification_report(y_te, y_pred_tuned, target_names=['No Closure', 'Closure']))
print(f"Road Closure ROC-AUC: {roc_auc_score(y_te, probs_c):.4f}")


# 7C. Regression — Duration Prediction (minutes)
# Uses FEATURES_ALL. Only rows with a known duration are included.
print("\n--- 7C. Duration Regression ---")
dur_mask_tr = df_train['duration_minutes'].notna()
dur_mask_te = df_test['duration_minutes'].notna()
X_tr_d = prep_X(df_train.loc[dur_mask_tr], FEATURES_ALL)
X_te_d = prep_X(df_test.loc[dur_mask_te],  FEATURES_ALL)
y_tr_d = np.log1p(df_train.loc[dur_mask_tr, 'duration_minutes'])
y_te_d = np.log1p(df_test.loc[dur_mask_te,  'duration_minutes'])

dur_pipe = SkPipeline([
    ('te', ce.TargetEncoder(cols=[c for c in target_enc_cols if c in X_tr_d.columns], smoothing=5)),
    ('reg', LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1, objective='quantile', alpha=0.5))
])
dur_pipe.fit(X_tr_d, y_tr_d)
y_pred_d = dur_pipe.predict(X_te_d)

# All metrics are computed on the ORIGINAL scale (minutes) by back-transforming
# via expm1(). Previously r2_score used the raw log-space arrays (y_te_d / y_pred_d)
# which made R² uninterpretable — a negative R² looked like "no skill" but was just
# an artefact of the log-scale residual geometry.
y_te_d_exp   = np.expm1(y_te_d)     # actual durations in minutes
y_pred_d_exp = np.expm1(y_pred_d)   # predicted durations in minutes

mae    = mean_absolute_error(y_te_d_exp, y_pred_d_exp)          # mean absolute error (minutes)
medae  = float(np.median(np.abs(y_te_d_exp - y_pred_d_exp)))    # median AE — robust to skew
r2     = r2_score(y_te_d_exp, y_pred_d_exp)                     # R² on original scale ← Fix 6
r2_log = r2_score(y_te_d,     y_pred_d)                         # log-space R² kept for reference
print(f"Duration Regression (original scale):")
print(f"  MAE={mae:.1f} min | MedAE={medae:.1f} min | R²={r2:.4f}  (log-space R²={r2_log:.4f})")


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


# ─── STEP 9: PRODUCTION LOOKUP TABLES ────────────────────────
# geo_stats, zone_hour_counts, and corridor_risk were all computed on
# df_train in Step 6.6 — they are safe to expose as production lookups.
print("\n--- Step 9: Creating Production Lookups for Dashboard ---")
geohash_lookup       = geo_stats.set_index('geohash6').to_dict(orient='index')
zone_hour_lookup     = zone_hour_counts.set_index(['zone', 'hour']).to_dict(orient='index')
corridor_risk_lookup = corridor_risk.set_index('corridor').to_dict(orient='index')
cause_closure_lookup = cause_closure.set_index('event_cause')['cause_closure_rate'].to_dict()
cause_duration_lookup= cause_duration.set_index('event_cause')['cause_avg_duration'].to_dict()

# global_medians come from geo_train_medians (train-derived, computed in Step 6.6)
global_medians = geo_train_medians


# ─── STEP 10: MODEL PERSISTENCE ───────────────────────────────
print("\n--- Step 10: Persisting Models & Metadata ---")
os.makedirs('models', exist_ok=True)

import json
eval_metrics = {
    "priority_roc_auc": round(float(roc_auc_score(y_test_p, y_prob)), 4),
    "priority_f1_weighted": round(float(lgbm_f1_cv), 4),
    "closure_roc_auc": round(float(roc_auc_score(y_te, probs_c)), 4),
    "closure_best_threshold": round(float(best_thresh), 3),
    "duration_mae": round(float(mae), 1),
    "duration_medae": round(float(medae), 1),
    "duration_r2": round(float(r2), 4),
    "priority_feature_importance": priority_feature_importance
}
with open("models/eval_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=2)

joblib.dump(final_pipe_priority, 'models/priority_classifier.pkl')
joblib.dump(final_pipe_closure, 'models/closure_classifier.pkl')
joblib.dump(dur_pipe, 'models/duration_regressor.pkl')
joblib.dump(db, 'models/dbscan_clusterer.pkl')
joblib.dump(cluster_profile, 'models/cluster_profiles.pkl')
joblib.dump(df_cluster, 'models/cluster_points.pkl')
joblib.dump(FEATURES_PRIORITY,    'models/feature_list_priority.pkl')  # priority model
joblib.dump(FEATURES_CLOSURE,     'models/feature_list_closure.pkl')   # closure model
joblib.dump(FEATURES_ALL,         'models/feature_list.pkl')           # duration model
joblib.dump(preprocessor_state,   'models/preprocessor.pkl')           # fitted IQR bounds + mode values + time_dummies

# ─── STEP 10.5: XAI GLOBAL SHAP EXPLANATIONS ────────────────────────
print("\n--- Step 10.5: Computing Global SHAP Explanations ---")
try:
    from xai_engine import (
        extract_base_model, transform_data, compute_shap_values,
        generate_global_importance_plot
    )
    
    # 1. Priority Model SHAP
    print("Computing SHAP for Priority Classifier...")
    # Sample 200 rows for background
    bg_p = df_test.sample(n=min(200, len(df_test)), random_state=42)
    X_bg_p = prep_X(bg_p, FEATURES_PRIORITY)
    X_bg_p_transformed = transform_data(final_pipe_priority, X_bg_p)
    model_p = extract_base_model(final_pipe_priority)
    _, shap_vals_p = compute_shap_values(model_p, X_bg_p_transformed)
    fig_p = generate_global_importance_plot(shap_vals_p, FEATURES_PRIORITY, title="Global Feature Importance - Priority")
    fig_p.write_html('models/xai_global_priority.html')
    
    # 2. Closure Model SHAP
    print("Computing SHAP for Closure Classifier...")
    bg_c = df_test.sample(n=min(200, len(df_test)), random_state=42)
    X_bg_c = prep_X(bg_c, FEATURES_CLOSURE)
    X_bg_c_transformed = transform_data(final_pipe_closure, X_bg_c)
    model_c = extract_base_model(final_pipe_closure)
    _, shap_vals_c = compute_shap_values(model_c, X_bg_c_transformed)
    fig_c = generate_global_importance_plot(shap_vals_c, FEATURES_CLOSURE, title="Global Feature Importance - Road Closure")
    fig_c.write_html('models/xai_global_closure.html')
    
    # 3. Duration Model SHAP
    print("Computing SHAP for Duration Regressor...")
    bg_d = df_test.loc[dur_mask_te].sample(n=min(200, dur_mask_te.sum()), random_state=42)
    X_bg_d = prep_X(bg_d, FEATURES_ALL)
    X_bg_d_transformed = transform_data(dur_pipe, X_bg_d)
    model_d = extract_base_model(dur_pipe)
    _, shap_vals_d = compute_shap_values(model_d, X_bg_d_transformed)
    fig_d = generate_global_importance_plot(shap_vals_d, FEATURES_ALL, title="Global Feature Importance - Event Duration")
    fig_d.write_html('models/xai_global_duration.html')
    
    print("XAI Global Plots saved to models/ as HTML files.")
except Exception as e:
    print(f"Failed to generate SHAP explanations: {e}")

# Dump lookups for Streamlit App
joblib.dump(geohash_lookup, 'models/geohash_lookup.pkl')
joblib.dump(zone_hour_lookup, 'models/zone_hour_lookup.pkl')
joblib.dump(corridor_risk_lookup, 'models/corridor_risk_lookup.pkl')
joblib.dump(cause_closure_lookup, 'models/cause_closure_lookup.pkl')
joblib.dump(cause_duration_lookup, 'models/cause_duration_lookup.pkl')
joblib.dump(global_medians, 'models/global_medians.pkl')
joblib.dump(best_thresh, 'models/closure_best_threshold.pkl')

print("All models, lookup tables, and configuration files saved successfully to models/ directory.")
print("Training Pipeline Done!")
