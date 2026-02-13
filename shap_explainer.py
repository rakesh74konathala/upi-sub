# shap_explainer.py
"""
SHAP Explainer for UPI Fraud Detection System
Provides interpretable explanations for XGBoost fraud predictions
"""

import pickle
import pandas as pd
import numpy as np
import shap
import os
from pathlib import Path


class FraudSHAPExplainer:
    """
    SHAP-based explainer for fraud detection model predictions
    """
    
    def __init__(self, model_dir='/teamspace/studios/this_studio/Dataset/models'):
        """
        Initialize SHAP explainer with pre-trained model components
        
        Args:
            model_dir (str): Path to directory containing saved model files
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.explainer = None
        self.metadata = None
        self._load_components()
        self._initialize_shap()
    
    def _load_components(self):
        """Load saved model components from pickle files"""
        try:
            # Load XGBoost model
            with open(self.model_dir / 'xgboost_model_clean.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load feature scaler
            with open(self.model_dir / 'feature_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load clean feature names
            with open(self.model_dir / 'clean_ml_features.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Load metadata
            with open(self.model_dir / 'complete_model_metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
            
            print("✅ All model components loaded successfully")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model component not found: {e}")
        except Exception as e:
            raise Exception(f"Error loading model components: {e}")
    
    def _initialize_shap(self):
        """Initialize SHAP explainer for XGBoost model"""
        try:
            # Create SHAP explainer (TreeExplainer for XGBoost)
            self.explainer = shap.TreeExplainer(self.model)
            print("✅ SHAP TreeExplainer initialized")
            
        except Exception as e:
            print(f"⚠️ SHAP explainer initialization failed: {e}")
            print("📝 Using fallback feature importance method")
            self.explainer = None
    
    def predict_with_explanation(self, transaction_data, threshold=0.5):
        """
        Predict fraud probability and generate explanation
        
        Args:
            transaction_data (dict): Transaction features as key-value pairs
            threshold (float): Classification threshold (default 0.5)
        
        Returns:
            dict: Prediction results with explanations
        """
        if not all([self.model, self.scaler, self.feature_names]):
            return {
                'error': "Model components not loaded. Cannot proceed with prediction.",
                'fraud_probability': None,
                'is_fraud': None
            }
        
        # Assertions to satisfy static type checker
        assert self.model is not None
        assert self.scaler is not None
        assert self.feature_names is not None

        try:
            # Prepare transaction data
            df = pd.DataFrame([transaction_data])
            
            # Scale features
            df_scaled = pd.DataFrame(
                self.scaler.transform(df[self.feature_names]),
                columns=self.feature_names
            )
            
            # Get prediction
            fraud_probability = self.model.predict_proba(df_scaled)[0][1]
            is_fraud = fraud_probability > threshold
            
            # Generate SHAP explanation
            explanation = self._generate_explanation(df_scaled.iloc[0], fraud_probability)
            
            return {
                'fraud_probability': float(fraud_probability),
                'is_fraud': bool(is_fraud),
                'risk_level': self._get_risk_level(fraud_probability),
                'explanation': explanation,
                'top_factors': explanation.get('top_contributing_factors', []),
                'recommendation': self._get_recommendation(fraud_probability, is_fraud)
            }
            
        except Exception as e:
            return {
                'error': f"Prediction failed: {str(e)}",
                'fraud_probability': None,
                'is_fraud': None
            }
    
    def _generate_explanation(self, scaled_transaction, fraud_probability):
        """Generate detailed SHAP-based explanation"""
        
        if self.explainer is not None:
            try:
                # Get SHAP values for this transaction
                shap_values = self.explainer.shap_values(scaled_transaction.values.reshape(1, -1))[0]
                
                # Assert feature_names is not None for type checker
                assert self.feature_names is not None

                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'value': scaled_transaction.values,
                    'shap_value': shap_values,
                    'abs_shap': np.abs(shap_values)
                }).sort_values('abs_shap', ascending=False)
                
                # Get top contributing factors
                top_factors = self._interpret_top_factors(feature_importance.head(5))
                
                base_value = self._get_safe_expected_value()
                prediction_value = base_value + np.sum(shap_values)

                return {
                    'method': 'SHAP',
                    'base_value': base_value,
                    'prediction_value': prediction_value,
                    'top_contributing_factors': top_factors,
                    'feature_impacts': feature_importance.head(10).to_dict('records')
                }
                
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
                return self._fallback_explanation(scaled_transaction, fraud_probability)
        else:
            return self._fallback_explanation(scaled_transaction, fraud_probability)
    
    def _fallback_explanation(self, scaled_transaction, fraud_probability):
        """Fallback explanation using feature importance"""
        
        # Use model's feature importance
        if self.model is None:
            return {
                'method': 'Feature Importance',
                'error': 'Model not available for feature importance calculation.',
                'top_contributing_factors': [],
                'feature_impacts': []
            }
            
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'value': scaled_transaction.values,
            'importance': self.model.feature_importances_,
            'weighted_impact': scaled_transaction.values * self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_factors = self._interpret_top_factors(feature_importance.head(5))
        
        return {
            'method': 'Feature Importance',
            'top_contributing_factors': top_factors,
            'feature_impacts': feature_importance.head(10).to_dict('records')
        }
    
    def _get_safe_expected_value(self) -> float:
        """Safely get the SHAP explainer's expected value as a float."""
        if self.explainer is None:
            return 0.0
        
        expected_value = self.explainer.expected_value
        
        if isinstance(expected_value, list):
            expected_value = expected_value[0]

        if expected_value is None:
            return 0.0
            
        try:
            return float(expected_value)  # type: ignore
        except (TypeError, ValueError):
            return 0.0

    def _interpret_top_factors(self, top_features_df):
        """Convert top features into human-readable explanations"""
        
        explanations = []
        
        for _, row in top_features_df.iterrows():
            feature = row['feature']
            value = row['value'] if 'value' in row else 0
            
            # Feature-specific interpretations
            explanation = self._get_feature_explanation(feature, value)
            
            explanations.append({
                'feature': feature,
                'explanation': explanation,
                'impact': 'HIGH' if abs(value) > 1.5 else 'MEDIUM' if abs(value) > 0.5 else 'LOW'
            })
        
        return explanations
    
    def _get_feature_explanation(self, feature, value):
        """Get human-readable explanation for specific features"""
        
        feature_explanations = {
            'orig_emptied': 'Account was completely emptied' if value > 0 else 'Account balance remained',
            'balance_amount_diff': 'Unusual balance change pattern' if abs(value) > 1 else 'Normal balance change',
            'amount': 'Large transaction amount' if value > 1 else 'Small transaction amount' if value < -1 else 'Typical amount',
            'is_night': 'Transaction during unusual hours' if value > 0 else 'Transaction during normal hours',
            'is_weekend': 'Weekend transaction' if value > 0 else 'Weekday transaction',
            'is_round_1000': 'Round amount (suspicious pattern)' if value > 0 else 'Non-round amount',
            'user_amount_cummean': 'Unusual for this user' if abs(value) > 1 else 'Typical for this user',
            'step': 'Late in time period' if value > 0 else 'Early in time period',
            'orig_zero_before': 'Account was empty before transaction' if value > 0 else 'Account had balance',
            'amount_vs_user_avg': 'Much larger than user average' if value > 1 else 'Smaller than user average' if value < -1 else 'Typical for user'
        }
        
        return feature_explanations.get(feature, f'{feature}: {"High" if value > 0 else "Low"} value')
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability >= 0.8:
            return 'VERY HIGH'
        elif probability >= 0.6:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        elif probability >= 0.2:
            return 'LOW'
        else:
            return 'VERY LOW'
    
    def _get_recommendation(self, probability, is_fraud):
        """Get recommended action based on prediction"""
        
        if probability >= 0.9:
            return "🚨 BLOCK TRANSACTION - Very high fraud risk. Require manual review."
        elif probability >= 0.7:
            return "⚠️ ADDITIONAL VERIFICATION - Require OTP or biometric confirmation."
        elif probability >= 0.5:
            return "🔍 MONITOR CLOSELY - Allow but flag for review."
        elif probability >= 0.3:
            return "📊 TRACK PATTERN - Allow transaction, monitor user behavior."
        else:
            return "✅ APPROVE - Low risk, process normally."
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.metadata:
            xgb_info = self.metadata.get('models', {}).get('xgboost_clean', {})
            return {
                'model_type': 'XGBoost Classifier',
                'features_count': len(self.feature_names) if self.feature_names else 0,
                'performance': xgb_info.get('performance', {}),
                'training_info': f"Trained on {xgb_info.get('training_samples', 'Unknown')} samples"
            }
        return {'model_type': 'XGBoost Classifier', 'features_count': len(self.feature_names) if self.feature_names else 0}


# Example usage and testing
if __name__ == "__main__":
    # Initialize explainer
    explainer = FraudSHAPExplainer()
    
    # Example transaction (replace with actual feature values)
    sample_transaction = {
        'step': 100,
        'amount': 50000,
        'balance_amount_diff': -50000,
        'hour_of_day': 2,  # 2 AM
        'day_of_week': 6,  # Weekend
        'is_weekend': 1,
        'is_night': 1,
        'amount_log': 10.82,
        'is_round_100': 1,
        'is_round_1000': 0,
        'balance_amount_ratio': 0.5,
        'orig_zero_before': 0,
        'orig_zero_after': 1,  # Account emptied
        'dest_zero_before': 1,
        'dest_zero_after': 0,
        'orig_emptied': 1,  # Red flag!
        'user_transaction_count': 1,
        'is_high_volume_user': 0,
        'time_since_last': 0,
        'user_trans_cumcount': 1,
        'user_amount_cummean': 50000,
        'amount_vs_user_avg': 1.0,
        'is_large_transaction': 1,
        'is_first_transaction': 1,
        'is_first_to_dest': 1,
        'dest_risk_score': 0,
        'is_high_risk_dest': 0
    }
    
    # Get prediction with explanation
    result = explainer.predict_with_explanation(sample_transaction)
    
    print("🧪 SHAP Explainer Test Results:")
    if 'error' in result and result['error']:
        print(f"⚠️ An error occurred: {result['error']}")
    else:
        print(f"Fraud Probability: {result.get('fraud_probability', 0.0):.3f}")
        print(f"Risk Level: {result.get('risk_level', 'N/A')}")
        print(f"Recommendation: {result.get('recommendation', 'N/A')}")
        print("\nTop Contributing Factors:")
        top_factors = result.get('top_factors', [])
        if top_factors:
            for factor in top_factors:
                print(f"  • {factor.get('explanation', 'N/A')} ({factor.get('impact', 'N/A')} impact)")
        else:
            print("  • No contributing factors available.")