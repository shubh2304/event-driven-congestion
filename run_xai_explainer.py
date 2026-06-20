import os
import joblib
import pandas as pd
from xai_engine import explain_single_prediction

def main():
    print("Loading models and feature lists...")
    try:
        pipe_priority = joblib.load('models/priority_classifier.pkl')
        pipe_closure = joblib.load('models/closure_classifier.pkl')
        pipe_duration = joblib.load('models/duration_regressor.pkl')
        
        features_p = joblib.load('models/feature_list_priority.pkl')
        features_c = joblib.load('models/feature_list_closure.pkl')
        features_d = joblib.load('models/feature_list.pkl')
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run train_pipeline.py first!")
        return

    # Load data to get a sample
    print("Loading sample data...")
    df = pd.read_csv("Astram_event_data_anonymized.csv")
    
    # Simple mock processing for a sample row just for demonstration
    # In a real scenario, this would go through the full prep_X pipeline
    # For now, we just pass the raw features and let the pipeline handle it
    
    # Find an accident event to explain
    sample_event = df[df['event_cause'] == 'accident'].iloc[0:1].copy()
    
    # We need to minimally prep the sample so it matches what the model expects
    # In practice, you would use the exact prep_X function from train_pipeline.py
    # Here we just manually fill the features for demonstration
    
    # Dummy prep to match feature columns
    for col in features_c:
        if col not in sample_event.columns:
            sample_event[col] = 0
            
    X_p = sample_event[features_p]
    X_c = sample_event[features_c]
    X_d = sample_event[features_d]
    
    print("\n" + "="*50)
    print("XAI EXPLANATION DEMO")
    print("="*50)
    print(f"Event: {sample_event['event_cause'].values[0]} at {sample_event['latitude'].values[0]}, {sample_event['longitude'].values[0]}")
    
    # 1. Explain Priority
    print("\n--- 1. PRIORITY MODEL ---")
    text_p, fig_p = explain_single_prediction(pipe_priority, X_p, features_p, is_regression=False, title="Why is Priority High/Low?")
    print(text_p)
    fig_p.write_html("waterfall_priority.html")
    
    # 2. Explain Closure
    print("\n--- 2. CLOSURE MODEL ---")
    text_c, fig_c = explain_single_prediction(pipe_closure, X_c, features_c, is_regression=False, title="Why will the road close?")
    print(text_c)
    fig_c.write_html("waterfall_closure.html")
    
    # 3. Explain Duration
    print("\n--- 3. DURATION MODEL ---")
    text_d, fig_d = explain_single_prediction(pipe_duration, X_d, features_d, is_regression=True, title="Why will it take this long?")
    print(text_d)
    fig_d.write_html("waterfall_duration.html")
    
    print("\n" + "="*50)
    print("Plots saved as HTML files: waterfall_priority.html, waterfall_closure.html, waterfall_duration.html")
    print("="*50)

if __name__ == "__main__":
    main()
