import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

def extract_base_model(pipe):
    """Extracts the underlying XGBoost/LightGBM model from the ImbPipeline/SkPipeline."""
    if hasattr(pipe, 'named_steps'):
        if 'clf' in pipe.named_steps:
            return pipe.named_steps['clf']
        if 'reg' in pipe.named_steps:
            return pipe.named_steps['reg']
    return pipe

def transform_data(pipe, X):
    """Applies preprocessing steps (like TargetEncoder) from the pipeline to X."""
    X_transformed = X.copy()
    if hasattr(pipe, 'named_steps') and 'te' in pipe.named_steps:
        # TargetEncoder needs to be transformed
        X_transformed = pipe.named_steps['te'].transform(X_transformed)
    return X_transformed

def compute_shap_values(model, X_transformed):
    """Computes SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    # Handle LightGBM/XGBoost formatting differences
    # Binary classification often returns a list [negative_class, positive_class]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    expected_value = explainer.expected_value
    # Ensure expected_value is a scalar
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[-1] if len(expected_value) > 1 else expected_value[0]
        
    return expected_value, shap_values

def generate_global_importance_plot(shap_values, feature_names, title="Global Feature Importance"):
    """Generates a Plotly bar chart for global feature importance (mean absolute SHAP)."""
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame for plotting
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values(by='Importance', ascending=True)
    
    # Plotly Bar Chart
    fig = px.bar(
        df_imp.tail(20),  # Show top 20 features
        x='Importance', 
        y='Feature', 
        orientation='h',
        title=title,
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        template='plotly_dark',
        xaxis_title="Mean |SHAP Value| (Impact on Model Output)"
    )
    return fig

def generate_waterfall_plot(expected_value, shap_values_single, feature_names, X_single, title="Prediction Explanation"):
    """Generates a Plotly waterfall plot for a single prediction."""
    feature_values = X_single.values[0] if isinstance(X_single, pd.DataFrame) else X_single
    
    df_shap = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values_single,
        'Feature Value': feature_values
    })
    
    # Format labels (e.g. "feature = value")
    df_shap['Label'] = df_shap.apply(lambda row: f"{row['Feature']} = {row['Feature Value']}", axis=1)
    
    # Sort by absolute SHAP value to get the top pushers
    df_shap['Abs_SHAP'] = df_shap['SHAP Value'].abs()
    top_features = df_shap.sort_values(by='Abs_SHAP', ascending=False).head(10)
    
    # Re-sort the top features by SHAP value for the waterfall visual flow
    top_features = top_features.sort_values(by='SHAP Value', ascending=True)
    
    # Create the waterfall
    fig = go.Figure(go.Waterfall(
        name="20", orientation="h",
        measure=["relative"] * len(top_features),
        y=top_features['Label'],
        x=top_features['SHAP Value'],
        connector={"line":{"color":"rgb(63, 63, 63)"}},
        decreasing={"marker":{"color":"#10b981"}}, # Green for negative impact
        increasing={"marker":{"color":"#ef4444"}}, # Red for positive impact
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        template='plotly_dark',
        xaxis_title="SHAP Value (Contribution to Prediction)",
        yaxis_title="Top Features"
    )
    return fig

def format_explanation_text(shap_values_single, feature_names, X_single, is_regression=False):
    """Generates a text summary of the top contributing factors."""
    feature_values = X_single.values[0] if isinstance(X_single, pd.DataFrame) else X_single
    
    df_shap = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values_single,
        'Feature Value': feature_values
    })
    
    df_shap['Abs_SHAP'] = df_shap['SHAP Value'].abs()
    top_pushers = df_shap.sort_values(by='Abs_SHAP', ascending=False).head(5)
    
    explanation = "### Top Factors Influencing this Prediction:\n"
    for _, row in top_pushers.iterrows():
        direction = "increased" if row['SHAP Value'] > 0 else "decreased"
        risk_type = "duration" if is_regression else "risk probability"
        
        # Simple formatting (SHAP values are log-odds for classifiers)
        impact = abs(row['SHAP Value'])
        impact_str = f"{impact:.3f}"
        
        explanation += f"- **{row['Feature']}** (value: {row['Feature Value']}) {direction} the {risk_type} by {impact_str}.\n"
        
    return explanation

def explain_single_prediction(pipe, X_single, feature_names, is_regression=False, title="Prediction Explanation"):
    """Wrapper to get text and plots for a single prediction."""
    model = extract_base_model(pipe)
    X_transformed = transform_data(pipe, X_single)
    expected_value, shap_values = compute_shap_values(model, X_transformed)
    
    shap_vals_single = shap_values[0] if shap_values.ndim > 1 else shap_values
    
    text_expl = format_explanation_text(shap_vals_single, feature_names, X_single, is_regression)
    waterfall_fig = generate_waterfall_plot(expected_value, shap_vals_single, feature_names, X_single, title=title)
    
    return text_expl, waterfall_fig
