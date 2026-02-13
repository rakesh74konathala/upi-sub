
# Model Loading Script for Production
import pickle
import pandas as pd
import numpy as np

def load_fraud_detection_models():
    """Load trained fraud detection models and components"""
    
    model_dir = "/teamspace/studios/this_studio/Dataset/models"
    
    # Load XGBoost model (recommended)
    with open(model_dir + "/xgboost_model_clean.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    
    # Load feature scaler
    with open(model_dir + "/feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # Load clean features list
    with open(model_dir + "/clean_ml_features.pkl", "rb") as f:
        clean_features = pickle.load(f)
    
    # Load metadata
    with open(model_dir + "/complete_model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    return xgb_model, scaler, clean_features, metadata

def predict_fraud(transaction_data, xgb_model, scaler, clean_features):
    """
    Predict fraud for a single transaction
    
    Args:
        transaction_data: dict with transaction features
        xgb_model: trained XGBoost model
        scaler: fitted StandardScaler
        clean_features: list of feature names
    
    Returns:
        fraud_probability: float (0-1)
        is_fraud_prediction: boolean
    """
    
    # Convert to DataFrame and scale
    df = pd.DataFrame([transaction_data])
    df_scaled = pd.DataFrame(scaler.transform(df[clean_features]), 
                            columns=clean_features)
    
    # Predict
    fraud_probability = xgb_model.predict_proba(df_scaled)[0][1]
    is_fraud = fraud_probability > 0.5  # Default threshold
    
    return fraud_probability, is_fraud

# Example usage:
# xgb_model, scaler, features, metadata = load_fraud_detection_models()
# prob, is_fraud = predict_fraud(transaction_dict, xgb_model, scaler, features)
