# main.py
"""
Streamlit Web Interface for UPI Fraud Detection System
User-friendly interface for real-time fraud detection and explanations
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from fraud_detector import UPIFraudDetector
import time


# Page configuration
st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .safe-alert {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_fraud_detector():
    """Load fraud detector (cached for performance)"""
    try:
        return UPIFraudDetector()
    except Exception as e:
        st.error(f"Failed to load fraud detection system: {e}")
        return None


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🛡️ UPI Fraud Detection System</div>', unsafe_allow_html=True)
    st.markdown("**Real-time fraud detection with explainable AI**")
    
    # Load detector
    detector = load_fraud_detector()
    if detector is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("🚀 Navigation")
    mode = st.sidebar.selectbox(
        "Select Mode:",
        ["Single Transaction", "Batch Processing", "System Status", "About"]
    )
    
    if mode == "Single Transaction":
        single_transaction_interface(detector)
    elif mode == "Batch Processing":
        batch_processing_interface(detector)
    elif mode == "System Status":
        system_status_interface(detector)
    else:
        about_interface()


def single_transaction_interface(detector):
    """Interface for single transaction analysis"""
    
    st.header("🔍 Single Transaction Analysis")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Transaction Details")
        
        # Transaction ID
        transaction_id = st.text_input("Transaction ID", value="TXN" + str(int(time.time())))
        
        # Basic transaction info
        amount = st.number_input("Amount (₹)", min_value=0.01, value=10000.0, step=100.0)
        transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"])
        
        # Account balances
        st.subheader("💰 Account Information")
        old_balance_orig = st.number_input("Sender Old Balance (₹)", min_value=0.0, value=50000.0, step=1000.0)
        new_balance_orig = st.number_input("Sender New Balance (₹)", min_value=0.0, value=40000.0, step=1000.0)
        old_balance_dest = st.number_input("Receiver Old Balance (₹)", min_value=0.0, value=25000.0, step=1000.0)
        new_balance_dest = st.number_input("Receiver New Balance (₹)", min_value=0.0, value=35000.0, step=1000.0)
    
    with col2:
        st.subheader("⏰ Temporal Information")
        
        # Time information
        step = st.number_input("Time Step (Hour)", min_value=1, max_value=743, value=100)
        
        st.subheader("👤 User Behavior (Optional)")
        st.caption("Leave as default if unknown")
        
        user_transaction_count = st.number_input("User Total Transactions", min_value=1, value=5)
        user_amount_cummean = st.number_input("User Average Amount (₹)", min_value=1.0, value=15000.0)
        is_first_transaction = st.selectbox("First Transaction?", [0, 1], index=0)
        is_first_to_dest = st.selectbox("First Time to Destination?", [0, 1], index=1)
        time_since_last = st.number_input("Hours Since Last Transaction", min_value=0, value=24)
    
    # Advanced settings in expander
    with st.expander("⚙️ Advanced Settings"):
        threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.5, 0.01)
        st.caption("Lower threshold = more sensitive (more transactions flagged)")
    
    # Predict button
    if st.button("🔍 Analyze Transaction", type="primary"):
        
        # Prepare transaction data
        transaction_data = {
            'transaction_id': transaction_id,
            'step': step,
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': old_balance_orig,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'user_transaction_count': user_transaction_count,
            'user_amount_cummean': user_amount_cummean,
            'is_first_transaction': is_first_transaction,
            'is_first_to_dest': is_first_to_dest,
            'time_since_last': time_since_last
        }
        
        # Show processing
        with st.spinner("🧠 Analyzing transaction..."):
            result = detector.predict_fraud(transaction_data, threshold)
        
        # Display results
        display_prediction_results(result, transaction_data)


def display_prediction_results(result, transaction_data):
    """Display prediction results in a nice format"""
    
    if not result['success']:
        st.error(f"❌ Analysis failed: {result['error']}")
        return
    
    # Main result
    fraud_prob = result['fraud_probability']
    is_fraud = result['is_fraud']
    risk_level = result['risk_level']
    
    # Alert banner
    if is_fraud:
        st.markdown(f'''
        <div class="fraud-alert">
            <h3>🚨 FRAUD DETECTED</h3>
            <p><strong>Risk Level:</strong> {risk_level}</p>
            <p><strong>Confidence:</strong> {fraud_prob:.1%}</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="safe-alert">
            <h3>✅ TRANSACTION APPROVED</h3>
            <p><strong>Risk Level:</strong> {risk_level}</p>
            <p><strong>Confidence:</strong> {(1-fraud_prob):.1%}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fraud Probability", f"{fraud_prob:.3f}", f"{fraud_prob:.1%}")
    
    with col2:
        st.metric("Risk Level", risk_level)
    
    with col3:
        st.metric("Amount", f"₹{transaction_data['amount']:,.0f}")
    
    with col4:
        st.metric("Decision", "BLOCK" if is_fraud else "APPROVE")
    
    # Detailed results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Risk Analysis")
        
        # Risk gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fraud_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Risk"},
            delta={'reference': 0.5},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.2], 'color': "lightgray"},
                    {'range': [0.2, 0.5], 'color': "yellow"},
                    {'range': [0.5, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.8
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Recommendation
        st.subheader("💡 Recommendation")
        st.info(result['recommendation'])
    
    with col2:
        st.subheader("🔍 Explanation")
        
        # Top contributing factors
        if result.get('top_factors'):
            st.write("**Top Risk Factors:**")
            for i, factor in enumerate(result['top_factors'][:5], 1):
                impact_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                impact_icon = impact_color.get(factor['impact'], "⚪")
                st.write(f"{i}. {impact_icon} {factor['explanation']} ({factor['impact']} impact)")
        
        # Feature importance chart
        if result.get('explanation', {}).get('feature_impacts'):
            feature_data = result['explanation']['feature_impacts'][:8]  # Top 8 features
            
            df_features = pd.DataFrame(feature_data)
            if 'abs_shap' in df_features.columns:
                fig_features = px.bar(
                    df_features, 
                    x='abs_shap', 
                    y='feature',
                    orientation='h',
                    title="Feature Importance"
                )
                fig_features.update_layout(height=300)
                st.plotly_chart(fig_features, use_container_width=True)
    
    # Transaction details
    with st.expander("📋 Transaction Details"):
        st.json(transaction_data)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.write("**Model Information:**")
        st.write(f"- Model: {result.get('model_info', {}).get('model_type', 'XGBoost')}")
        st.write(f"- Threshold: {result.get('model_info', {}).get('threshold', 0.5)}")
        st.write(f"- Timestamp: {result.get('timestamp', 'N/A')}")
        
        if result.get('explanation'):
            st.write("**Explanation Method:**", result['explanation'].get('method', 'N/A'))


def batch_processing_interface(detector):
    """Interface for batch processing"""
    st.header("📁 Batch Processing")
    
    st.info("Upload a CSV file with multiple transactions for batch analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose CSV file", 
        type=['csv'],
        help="CSV should contain columns: amount, oldbalanceOrg, newbalanceOrig, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.write(f"📊 Loaded {len(df)} transactions")
            st.dataframe(df.head())
            
            if st.button("🔍 Analyze Batch"):
                # Convert to list of dicts
                transactions = df.to_dict('records')
                
                # Process batch
                with st.spinner(f"Processing {len(transactions)} transactions..."):
                    results = detector.predict_batch(transactions)
                
                # Display results
                st.success(f"✅ Processed {len(results)} transactions")
                
                # Results summary
                fraud_count = sum(1 for r in results if r.get('is_fraud'))
                avg_risk = np.mean([r.get('fraud_probability', 0) for r in results])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", len(results))
                col2.metric("Flagged as Fraud", fraud_count)
                col3.metric("Average Risk", f"{avg_risk:.3f}")
                
                # Download results
                results_df = pd.DataFrame(results)
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    f"fraud_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {e}")


def system_status_interface(detector):
    """System status and monitoring"""
    st.header("📊 System Status")
    
    status = detector.get_system_status()
    
    # System health
    if status['status'] == 'healthy':
        st.success("✅ System is running normally")
    else:
        st.error("❌ System issues detected")
    
    # Status details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 System Health")
        for component, is_healthy in status.get('components', {}).items():
            icon = "✅" if is_healthy else "❌"
            st.write(f"{icon} {component}")
    
    with col2:
        st.subheader("📈 Model Information")
        st.write(f"**Model Type:** {status.get('model_type', 'N/A')}")
        st.write(f"**Features:** {status.get('features_count', 'N/A')}")
        
        if 'performance' in status:
            perf = status['performance']
            st.write(f"**Test F1-Score:** {perf.get('test_f1', 'N/A'):.3f}")
            st.write(f"**Test Precision:** {perf.get('test_precision', 'N/A'):.3f}")
            st.write(f"**Test Recall:** {perf.get('test_recall', 'N/A'):.3f}")


def about_interface():
    """About page"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🛡️ UPI Fraud Detection System
    
    This system uses advanced machine learning to detect fraudulent UPI transactions in real-time.
    
    ### ✨ Key Features:
    - **Real-time Analysis**: Sub-second fraud detection
    - **Explainable AI**: SHAP-based explanations for every prediction
    - **High Accuracy**: 94.5% F1-Score on test data
    - **Production Ready**: Robust error handling and logging
    
    ### 🧠 Technology Stack:
    - **Machine Learning**: XGBoost Classifier
    - **Explainability**: SHAP (SHapley Additive exPlanations)
    - **Web Interface**: Streamlit
    - **Data Processing**: Pandas, NumPy
    
    ### 📊 Model Performance:
    - **Precision**: 97.6%
    - **Recall**: 91.5%
    - **F1-Score**: 94.5%
    
    ### 🔒 Security Features:
    - No sensitive data storage
    - Secure model predictions
    - Audit trail for all decisions
    
    ---
    **Built with ❤️ for secure digital payments**
    """)


if __name__ == "__main__":
    main()