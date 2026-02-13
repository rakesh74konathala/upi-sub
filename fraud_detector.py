# fraud_detector.py
"""
Main Fraud Detection Engine for UPI Fraud Detection System
Orchestrates feature engineering, prediction, and explanation generation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from shap_explainer import FraudSHAPExplainer


class UPIFraudDetector:
    """
    Main fraud detection engine that processes raw transaction data
    and returns predictions with explanations
    """
    
    def __init__(self, model_dir='/teamspace/studios/this_studio/Dataset/models'):
        """
        Initialize fraud detection engine
        
        Args:
            model_dir (str): Path to saved model components
        """
        self.model_dir = model_dir
        self.explainer = None
        self._setup_logging()
        self._initialize_components()
    
    def _setup_logging(self):
        """Setup logging for the fraud detection system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize SHAP explainer and other components"""
        try:
            self.explainer = FraudSHAPExplainer(self.model_dir)
            self.logger.info("✅ Fraud detection engine initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize fraud detection engine: {e}")
            raise
    
    def engineer_features(self, raw_transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw transaction data into ML features
        
        Args:
            raw_transaction (dict): Raw transaction data from UPI system
            
        Returns:
            dict: Engineered features ready for model prediction
        """
        try:
            # Extract basic transaction info
            step = raw_transaction.get('step', 1)
            transaction_type = raw_transaction.get('type', 'PAYMENT')
            amount = float(raw_transaction.get('amount', 0))
            old_balance_orig = float(raw_transaction.get('oldbalanceOrg', 0))
            new_balance_orig = float(raw_transaction.get('newbalanceOrig', 0))
            old_balance_dest = float(raw_transaction.get('oldbalanceDest', 0))
            new_balance_dest = float(raw_transaction.get('newbalanceDest', 0))
            
            # 1. Basic Features
            features = {
                'step': step,
                'amount': amount,
                'balance_amount_diff': new_balance_orig - old_balance_orig
            }
            
            # 2. Temporal Features
            hour_of_day = step % 24
            day_of_week = (step // 24) % 7
            
            features.update({
                'hour_of_day': hour_of_day,
                'day_of_week': day_of_week,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'is_night': 1 if hour_of_day >= 22 or hour_of_day <= 6 else 0
            })
            
            # 3. Amount Features
            features.update({
                'amount_log': np.log1p(amount),
                'is_round_100': 1 if (amount % 100 == 0 and amount > 0) else 0,
                'is_round_1000': 1 if (amount % 1000 == 0 and amount > 0) else 0
            })
            
            # Amount category (0-4 based on amount ranges)
            if amount <= 100:
                amount_category = 0
            elif amount <= 1000:
                amount_category = 1
            elif amount <= 10000:
                amount_category = 2
            elif amount <= 100000:
                amount_category = 3
            else:
                amount_category = 4
            
            # 4. Balance Features
            features.update({
                'balance_amount_ratio': old_balance_orig / amount if amount > 0 else 0,
                'orig_zero_before': 1 if old_balance_orig == 0 else 0,
                'orig_zero_after': 1 if new_balance_orig == 0 else 0,
                'dest_zero_before': 1 if old_balance_dest == 0 else 0,
                'dest_zero_after': 1 if new_balance_dest == 0 else 0,
                'orig_emptied': 1 if (old_balance_orig > 0 and new_balance_orig == 0) else 0
            })
            
            # 5. User Behavior Features (simplified for single transaction)
            # In production, these would come from user history database
            features.update({
                'user_transaction_count': raw_transaction.get('user_transaction_count', 1),
                'is_high_volume_user': raw_transaction.get('is_high_volume_user', 0),
                'time_since_last': raw_transaction.get('time_since_last', 0),
                'user_trans_cumcount': raw_transaction.get('user_trans_cumcount', 1),
                'user_amount_cummean': raw_transaction.get('user_amount_cummean', amount),
                'amount_vs_user_avg': amount / raw_transaction.get('user_amount_cummean', amount) if raw_transaction.get('user_amount_cummean', amount) > 0 else 1,
                'is_large_transaction': 1 if amount > raw_transaction.get('user_amount_cummean', amount) * 3 else 0,
                'is_first_transaction': raw_transaction.get('is_first_transaction', 0),
                'is_first_to_dest': raw_transaction.get('is_first_to_dest', 1)
            })
            
            self.logger.info(f"✅ Features engineered for transaction amount: {amount}")
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {e}")
            raise ValueError(f"Feature engineering failed: {e}")
    
    def validate_transaction(self, raw_transaction: Dict[str, Any]) -> bool:
        """
        Validate raw transaction data
        
        Args:
            raw_transaction (dict): Raw transaction data
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
        
        for field in required_fields:
            if field not in raw_transaction:
                self.logger.error(f"❌ Missing required field: {field}")
                return False
            
            try:
                float(raw_transaction[field])
            except (ValueError, TypeError):
                self.logger.error(f"❌ Invalid numeric value for {field}: {raw_transaction[field]}")
                return False
        
        # Check amount is positive
        if float(raw_transaction['amount']) <= 0:
            self.logger.error(f"❌ Invalid amount: {raw_transaction['amount']}")
            return False
        
        return True
    
    def predict_fraud(self, raw_transaction: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
        """
        Main method to predict fraud for a transaction
        
        Args:
            raw_transaction (dict): Raw transaction data
            threshold (float): Classification threshold
            
        Returns:
            dict: Complete fraud prediction with explanations
        """
        try:
            # 1. Validate input
            if not self.validate_transaction(raw_transaction):
                return {
                    'success': False,
                    'error': 'Invalid transaction data',
                    'fraud_probability': None,
                    'is_fraud': None
                }
            
            # 2. Engineer features
            features = self.engineer_features(raw_transaction)
            
            # 3. Get prediction with explanation
            if self.explainer is None:
                return {
                    'success': False,
                    'error': 'SHAP explainer not initialized',
                    'fraud_probability': None,
                    'is_fraud': None
                }
            
            result = self.explainer.predict_with_explanation(features, threshold)
            
            # 4. Add metadata
            result.update({
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'transaction_id': raw_transaction.get('transaction_id', 'N/A'),
                'amount': raw_transaction['amount'],
                'model_info': {
                    'model_type': 'XGBoost',
                    'version': '1.0',
                    'threshold': threshold
                }
            })
            
            self.logger.info(f"✅ Prediction completed - Fraud prob: {result.get('fraud_probability', 0):.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fraud_probability': None,
                'is_fraud': None,
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(self, transactions: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Predict fraud for multiple transactions
        
        Args:
            transactions (list): List of raw transaction data
            threshold (float): Classification threshold
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i, transaction in enumerate(transactions):
            self.logger.info(f"Processing transaction {i+1}/{len(transactions)}")
            result = self.predict_fraud(transaction, threshold)
            results.append(result)
        
        self.logger.info(f"✅ Batch prediction completed for {len(transactions)} transactions")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system health and status"""
        status = {
            'system': 'UPI Fraud Detection Engine',
            'status': 'healthy' if self.explainer is not None else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'explainer_loaded': self.explainer is not None,
                'model_path': str(self.model_dir)
            }
        }
        
        if self.explainer:
            status.update(self.explainer.get_model_info())
        
        return status


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = UPIFraudDetector()
    
    # Example transaction data
    sample_transaction = {
        'transaction_id': 'TXN123456789',
        'step': 150,
        'type': 'TRANSFER',
        'amount': 75000,
        'oldbalanceOrg': 100000,
        'newbalanceOrig': 25000,
        'oldbalanceDest': 50000,
        'newbalanceDest': 125000,
        'user_transaction_count': 5,
        'user_amount_cummean': 20000,  # User typically does smaller amounts
        'is_first_transaction': 0,
        'is_first_to_dest': 1  # First time to this destination
    }
    
    # Predict fraud
    result = detector.predict_fraud(sample_transaction)
    
    print("🧪 Fraud Detection Engine Test:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Fraud Probability: {result['fraud_probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Decision: {'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"\nTop Risk Factors:")
        for factor in result.get('top_factors', [])[:3]:
            print(f"  • {factor['explanation']}")
    else:
        print(f"Error: {result['error']}")
    
    # Test system status
    status = detector.get_system_status()
    print(f"\n📊 System Status: {status['status']}")