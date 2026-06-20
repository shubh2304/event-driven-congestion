import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Common models load
models = {
    'priority':            joblib.load('models/priority_classifier.pkl'),
    'closure':             joblib.load('models/closure_classifier.pkl'),
    'duration':            joblib.load('models/duration_regressor.pkl'),
    'features_priority':   joblib.load('models/feature_list_priority.pkl'),
    'features_closure':    joblib.load('models/feature_list_closure.pkl'),
    'features':            joblib.load('models/feature_list.pkl'),
    'geohash_lookup':      joblib.load('models/geohash_lookup.pkl'),
    'zone_hour_lookup':    joblib.load('models/zone_hour_lookup.pkl'),
    'corridor_risk_lookup':joblib.load('models/corridor_risk_lookup.pkl'),
    'cause_closure_lookup':joblib.load('models/cause_closure_lookup.pkl'),
    'cause_duration_lookup':joblib.load('models/cause_duration_lookup.pkl'),
    'global_medians':      joblib.load('models/global_medians.pkl'),
    'closure_best_thresh': joblib.load('models/closure_best_threshold.pkl'),
    'preprocessor':        joblib.load('models/preprocessor.pkl'),
    'eval_metrics':        json.load(open('models/eval_metrics.json')),
}

try:
    import geohash2 as gh
    def encode_geohash(lat, lon, precision=6): return gh.encode(lat, lon, precision)
except:
    import pygeohash as pgh
    def encode_geohash(lat, lon, precision=6): return pgh.encode(lat, lon, precision)

# Event properties
LAT, LON = 12.9716, 77.5946
HOUR = 9
DAY = 0  # Monday
MONTH = 6
CAUSE = 'accident'
TYPE = 'unplanned'
VEH = 'others'
ZONE = 'Central Zone 2'
CORR = 'Non-corridor'
PS = 'Cubbon Park'

print("=== Cross-Interface Consistency Test ===")
print(f"Event: {CAUSE} at {LAT}, {LON} on Hour {HOUR}")
print()

# ---------------------------------------------------------
# 1. API Test
# ---------------------------------------------------------
from fastapi.testclient import TestClient
from api import app as fastapi_app

with TestClient(fastapi_app) as client:
    resp = client.post('/predict', json={
        'latitude': LAT, 'longitude': LON, 'hour': HOUR, 'day_of_week': DAY, 'month': MONTH,
        'event_cause': CAUSE, 'event_type': TYPE, 'veh_type': VEH, 'zone': ZONE, 'corridor': CORR,
        'police_station': PS
    })
    if resp.status_code == 200:
        data = resp.json()
        print("--- API (api.py) ---")
        print(f"Priority Risk : {data['priority_risk']:.4f}")
        print(f"Closure Risk  : {data['closure_risk']:.4f}")
        print(f"Duration Est  : {data['estimated_duration_min']} min")
    else:
        print("API Error:", resp.text)

print()

# ---------------------------------------------------------
# 2. Chatbot Test
# ---------------------------------------------------------
from chatbot import generate_predict_response
from app import generate_recommendations

params = {
    'latitude': LAT, 'longitude': LON, 'hour': HOUR, 'day_of_week': DAY, 'month': MONTH,
    'event_cause': CAUSE, 'veh_type': VEH, 'zone': ZONE, 'corridor': CORR, 'police_station': PS
}
chat_resp = generate_predict_response(params, models, encode_geohash, generate_recommendations)

import re
prio_match = re.search(r'Priority Risk.*?(\d+\.\d+)%', chat_resp)
close_match = re.search(r'Road Closure Risk.*?(\d+\.\d+)%', chat_resp)
dur_match = re.search(r'Estimated Duration\*\*.*?(\d+)\s+min', chat_resp)

print("--- CHATBOT (chatbot.py) ---")
print(f"Priority Risk : {float(prio_match.group(1))/100:.4f}" if prio_match else "Prio missing")
print(f"Closure Risk  : {float(close_match.group(1))/100:.4f}" if close_match else "Closure missing")
print(f"Duration Est  : {dur_match.group(1)} min" if dur_match else "Duration missing")

print()

# ---------------------------------------------------------
# 3. App.py Simulation (copy-paste of app.py inference block)
# ---------------------------------------------------------
import app as st_app
# We run the exact same logic block inside app.py predict button
day_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
hour_bins = [-1,5,11,16,20,24]
hour_labels = ['night','morning','afternoon','evening','late_evening']
tod = pd.cut([HOUR], bins=hour_bins, labels=hour_labels)[0]

geohash_lookup = models['geohash_lookup']
zone_hour_lookup = models['zone_hour_lookup']
corridor_risk_lookup = models['corridor_risk_lookup']
global_medians = models['global_medians']
closure_best_thresh = models['closure_best_thresh']

geohash_val = encode_geohash(LAT, LON, precision=6)

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

zh_key = (ZONE, HOUR)
if zh_key in zone_hour_lookup:
    zone_hour_event_count = zone_hour_lookup[zh_key].get('zone_hour_event_count', global_medians['zone_hour_event_count'])
else:
    zone_hour_event_count = global_medians['zone_hour_event_count']

corridor_key = CORR.strip().lower()
if corridor_key in corridor_risk_lookup:
    corridor_risk_score = corridor_risk_lookup[corridor_key].get('corridor_risk_score', global_medians['corridor_risk_score'])
else:
    corridor_risk_score = global_medians['corridor_risk_score']

if CAUSE in models['cause_closure_lookup']:
    cause_closure_rate = models['cause_closure_lookup'][CAUSE]
else:
    cause_closure_rate = global_medians['cause_closure_rate']

if CAUSE in models['cause_duration_lookup']:
    cause_avg_duration = models['cause_duration_lookup'][CAUSE]
else:
    cause_avg_duration = global_medians['cause_avg_duration']

input_dict = {
    'is_planned': int(TYPE == 'planned'),
    'hour': HOUR,
    'day_of_week': DAY,
    'month': MONTH,
    'is_weekend': int(DAY >= 5),
    'is_peak_hour': int(HOUR in [7,8,9,17,18,19,20]),
    'latitude': LAT,
    'longitude': LON,
    'event_cause': CAUSE,
    'veh_type': VEH,
    'corridor': CORR,
    'police_station': PS,
    'zone': ZONE,
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

prio_prob  = models['priority'].predict_proba(input_df_priority)[0][1]
close_prob = models['closure'].predict_proba(input_df_closure)[0][1]
dur_log = models['duration'].predict(input_df_all)[0]
dur_est = max(1.0, np.expm1(dur_log))

print("--- APP (app.py) ---")
print(f"Priority Risk : {prio_prob:.4f}")
print(f"Closure Risk  : {close_prob:.4f}")
print(f"Duration Est  : {round(dur_est)} min")
