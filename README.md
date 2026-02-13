# 🛡️ UPI Fraud Detection System: Real-Time AI Defense

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=for-the-badge&logo=streamlit)
![ML Model](https://img.shields.io/badge/XGBoost-1.7%2B-green?style=for-the-badge&logo=xgboost)
![Explainability](https://img.shields.io/badge/XAI-SHAP-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Maintained-Yes-brightgreen?style=for-the-badge)

## 📋 Table of Contents

1.  [Executive Summary](#-executive-summary)
2.  [The Challenge: Real-Time Payment Fraud](#-the-challenge-real-time-payment-fraud)
3.  [System Architecture & Workflow](#-system-architecture--workflow)
4.  [Key Features & Capabilities](#-key-features--capabilities)
5.  [Technical Deep Dive](#-technical-deep-dive)
    *   [Data Pipeline & Feature Engineering](#1-data-pipeline--feature-engineering)
    *   [Machine Learning Model (XGBoost)](#2-machine-learning-model-xgboost)
    *   [Explainable AI (SHAP)](#3-explainable-ai-shap)
6.  [Installation & Setup Guide](#-installation--setup-guide)
    *   [Prerequisites](#prerequisites)
    *   [Step-by-Step Installation](#step-by-step-installation)
    *   [Model Training (First Run)](#model-training-first-run)
7.  [User Manual](#-user-manual)
    *   [Single Transaction Analysis](#single-transaction-analysis)
    *   [Batch Processing](#batch-processing)
    *   [Interpreting Results](#interpreting-results)
8.  [Data Dictionary](#-data-dictionary)
9.  [Performance Benchmarks](#-performance-benchmarks)
10. [Troubleshooting & FAQ](#-troubleshooting--faq)
11. [Project Roadmap](#-project-roadmap)
12. [Contributing](#-contributing)
13. [License & Acknowledgements](#-license--acknowledgements)

---

## 🚀 Executive Summary

The **UPI Fraud Detection System** is an enterprise-grade, real-time security solution designed to combat the rising tide of digital payment fraud in Unified Payments Interface (UPI) ecosystems. Leveraging state-of-the-art Machine Learning (**XGBoost**) and Explainable AI (**SHAP**), this system provides sub-second classification of transactions as either **Legitimate** or **Fraudulent**.

Unlike "black box" AI solutions that provide a simple binary output, this project prioritizes **trust and transparency**. Every prediction is accompanied by a detailed, interpretable explanation, pinpointing exactly *why* a transaction was flagged—whether it was an unusual time of day, a suspicious amount pattern, or an anomaly in user behavior. This empowers fraud analysts to make informed decisions quickly, reducing false positives and improving customer trust.

---

## ⚠️ The Challenge: Real-Time Payment Fraud

Digital payments have revolutionized commerce, but they have also created new avenues for financial crime. The UPI ecosystem faces unique challenges:

*   **Velocity:** Transactions happen in milliseconds, requiring detection systems to be equally fast.
*   **Volume:** The sheer number of transactions makes manual review impossible.
*   **Complexity:** Fraud patterns evolve rapidly. Static rule-based systems (e.g., "block > $10k") are brittle and easily bypassed.
*   **Imbalance:** Genuine transactions vastly outnumber fraudulent ones (often 99.9% vs 0.1%), making it hard for standard models to learn fraud patterns without generating excessive false alarms.

This system addresses these challenges by using a gradient boosting model specifically tuned for imbalanced datasets, coupled with a feature engineering pipeline that captures behavioral context rather than just raw numbers.

---

## 🏗️ System Architecture & Workflow

The solution is built on a modular, microservices-ready architecture using Python.

### 1. The Frontend (Streamlit)
*   **`main.py`**: The interactive web dashboard. It handles user input, manages session state, and renders visualizations.
*   **UI Components**: Uses Plotly for gauge charts (risk score), bar charts (feature importance), and interactive data tables.

### 2. The Core Engine (`UPIFraudDetector`)
*   **`fraud_detector.py`**: The orchestration layer.
    *   **Input Validation**: Checks for missing fields, negative amounts, and data type mismatches.
    *   **Feature Engineering**: Transforms raw inputs (e.g., timestamp) into predictive features (e.g., "Is Night?").
    *   **Inference**: Passes the processed vector to the XGBoost model.
    *   **Health Checks**: Monitors system status and model availability.

### 3. The Explainer (`FraudSHAPExplainer`)
*   **`shap_explainer.py`**: The interpretability layer.
    *   **SHAP Calculation**: Uses TreeExplainer to calculate the marginal contribution of each feature.
    *   **Narrative Generation**: Converts numerical SHAP values into human-readable text (e.g., "High risk due to account emptying").

### 4. The Model Store (`Dataset/models`)
*   **Persistence**: Uses `pickle` to serialize trained artifacts.
    *   `xgboost_model_clean.pkl`: The trained classifier.
    *   `feature_scaler.pkl`: StandardScaler object for normalization.
    *   `clean_ml_features.pkl`: List of feature names to ensure alignment.
    *   `complete_model_metadata.pkl`: Performance metrics and training timestamp.

---

## ✨ Key Features & Capabilities

### 🛡️ Real-Time Detection
*   **Low Latency**: Optimized for < 200ms inference time per transaction.
*   **High Throughput**: Capable of processing batch files with thousands of rows in seconds.

### 🧠 Advanced Contextual Intelligence
The system goes beyond simple rules by understanding context:
*   **Temporal Analysis**: Detects anomalies based on time of day (e.g., high-value transfers at 3 AM).
*   **Behavioral Profiling**: Compares current transaction against user history (e.g., "Is this amount typical for this user?").
*   **Pattern Recognition**: Identifies synthetic patterns like "round number" fraud (e.g., 5000.00 vs 4932.12).

### 🔍 Explainable Decisions (XAI)
*   **Risk Scoring**: Provides a probabilistic score (0.00 - 1.00) rather than just a Yes/No.
*   **Factor Analysis**: Lists the top 5 distinct reasons for the risk score.
*   **Visual Proof**: Dynamic charts show how each feature pushed the prediction towards "Fraud" or "Safe".

### 📂 Robust Batch Processing
*   **Bulk Upload**: Support for CSV file uploads.
*   **Automated Reporting**: Generates a downloadable report with risk scores for every transaction.
*   **Aggregated Metrics**: Dashboard summary of total value at risk and fraud attack rate.

---

## 🔬 Technical Deep Dive

### 1. Data Pipeline & Feature Engineering
Raw transaction data is rarely predictive enough on its own. We transform 8 raw inputs into **25+ sophisticated features**:

#### A. Temporal Features
*   **`hour_of_day`**: Extracted from the simulation step (1 step = 1 hour).
*   **`is_night`**: Binary flag for high-risk hours (typically 12 AM - 6 AM).
*   **`is_weekend`**: Fraud patterns often shift on non-business days.

#### B. Amount Features
*   **`amount_log`**: Logarithmic transformation (`log(x+1)`) to handle the massive skew in transaction amounts (ranging from $1 to $10M).
*   **`is_round_amount`**: Fraudsters often use round numbers. We detect divisibility by 100 and 1000.
*   **`amount_vs_user_avg`**: Ratio of current amount to the user's historical average.

#### C. Account Balance Features
*   **`orig_emptied`**: Strong signal. Did the sender empty their account to exactly 0?
*   **`dest_zero_before`**: Was the destination account brand new (0 balance) before receiving funds?
*   **`balance_error`**: Does `oldBalance + amount` equal `newBalance`? Discrepancies often indicate system exploits.

### 2. Machine Learning Model (XGBoost)
*   **Algorithm**: **XGBoost (Extreme Gradient Boosting)** Classifier.
*   **Why XGBoost?**
    *   **Gradient Boosting**: Builds trees sequentially, correcting errors of previous trees.
    *   **Regularization**: Built-in L1/L2 regularization prevents overfitting on small fraud samples.
    *   **Speed**: Optimized for sparse data and parallel processing.
*   **Hyperparameters**:
    *   `scale_pos_weight`: Critical for imbalanced data. Weights the positive class (Fraud) higher to prevent the model from ignoring it.
    *   `max_depth`: Limited to prevent memorizing noise.
    *   `learning_rate`: Set low (e.g., 0.05) for robust generalization.

### 3. Explainable AI (SHAP)
We use **SHAP (SHapley Additive exPlanations)**, a game-theoretic approach to explain the output of any machine learning model.
*   **Global Importance**: Aggregates SHAP values to show which features matter most overall (e.g., "Transaction Type is the #1 predictor").
*   **Local Importance**: Shows individual feature contributions for a specific prediction.
    *   *Example*: For Transaction #123, `amount` added +20% risk, but `user_history` reduced risk by -5%.

---

## 💻 Installation & Setup Guide

### Prerequisites
*   **OS**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+).
*   **Python**: Version 3.8, 3.9, or 3.10.
*   **RAM**: Minimum 4GB (8GB recommended for training).

### Step-by-Step Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/upi-fraud-detection.git
    cd upi-fraud-detection
    ```

2.  **Create a Virtual Environment** (Highly Recommended)
    *   **Windows:**
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    *   **Mac/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *If `requirements.txt` is missing, install manually:*
    ```bash
    pip install streamlit pandas numpy xgboost shap plotly matplotlib seaborn scikit-learn jupyter
    ```

4.  **Verify Installation**
    ```bash
    streamlit --version
    python -c "import xgboost; print(xgboost.__version__)"
    ```

### Model Training (First Run)
The repository may not contain the trained model artifacts due to size limits. You must train the model locally first.

1.  **Download Data**: Download the PaySim dataset from Kaggle and place `PS_20174392719_1491204439457_log.csv` in the `Dataset/` folder.
2.  **Launch Notebook**:
    ```bash
    jupyter notebook "Upi fraud detection notebook.ipynb"
    ```
3.  **Run All Cells**: Execute the notebook to process data, train the model, and save artifacts to `Dataset/models/`.
4.  **Verify Artifacts**: Ensure `xgboost_model_clean.pkl` exists in `Dataset/models/`.

---

## 📖 User Manual

### Single Transaction Analysis
Ideal for analysts investigating a specific alert or testing scenarios.
1.  **Navigation**: Select "Single Transaction" from the sidebar.
2.  **Input Details**:
    *   **Transaction ID**: (Optional) For logging.
    *   **Amount**: The value of the transfer.
    *   **Transaction Type**: CASH_OUT and TRANSFER are highest risk.
    *   **Balances**: Enter old/new balances for Sender and Receiver.
3.  **Context**: (Optional) Input known user history (e.g., "Is this their first transaction?").
4.  **Threshold**: Adjust the sensitivity slider (default 0.5). Lower values flag more fraud but increase false alarms.
5.  **Analyze**: Click the button to run inference.

### Batch Processing
Ideal for auditing historical logs or processing daily dumps.
1.  **Navigation**: Select "Batch Processing".
2.  **Prepare CSV**: Create a CSV file with columns: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`.
3.  **Upload**: Drag and drop the file.
4.  **Process**: The system will engineer features for every row and run inference.
5.  **Export**: Download the result CSV, which includes a new `fraud_probability` column.

### Interpreting Results
*   **Risk Meter**: Green (Safe), Yellow (Caution), Red (High Risk).
*   **Prediction Confidence**: How sure is the model? (e.g., 99.8%).
*   **Top Risk Factors**: The specific reasons for the decision.
    *   *Example*: "1. High Impact: Receiver account was created just now."

---

## 📂 Data Dictionary

Understanding the input data is crucial for accurate analysis.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `step` | Integer | Maps to a unit of time (1 step = 1 hour). Total steps = 743 (30 days). |
| `type` | String | Type of transaction: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER. |
| `amount` | Float | Amount of the transaction in local currency. |
| `nameOrig` | String | Customer ID starting the transaction. |
| `oldbalanceOrg` | Float | Initial balance before the transaction. |
| `newbalanceOrig` | Float | New balance after the transaction. |
| `nameDest` | String | Customer ID receiving the transaction. |
| `oldbalanceDest` | Float | Initial balance recipient before transaction. |
| `newbalanceDest` | Float | New balance recipient after transaction. |
| `isFraud` | Boolean | (Target) 1 if fraud, 0 if legitimate. |

---

## 📊 Performance Benchmarks

The model was evaluated on a rigorous hold-out test set (20% of data).

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **99.9%** | Overall correctness. High because legitimate transactions dominate. |
| **Precision** | **97.6%** | When we say "Fraud", we are right 97.6% of the time. |
| **Recall** | **91.5%** | We catch 91.5% of all actual fraud attempts. |
| **F1-Score** | **94.5%** | The harmonic mean of Precision and Recall. The best single metric. |
| **AUC-ROC** | **0.99** | Excellent capability to separate classes. |

---

## ❓ Troubleshooting & FAQ

**Q: The app shows "Model not found" error.**
*   **A:** You haven't trained the model yet. Run the `Upi fraud detection notebook.ipynb` to generate the `.pkl` files in `Dataset/models/`.

**Q: I get a `ModuleNotFoundError: No module named 'shap'`.**
*   **A:** Install the missing library: `pip install shap`. On Windows, you might need to install C++ build tools if SHAP fails to compile.

**Q: The batch upload fails.**
*   **A:** Check your CSV format. It must strictly match the column names listed in the Data Dictionary. Dates should be converted to `step` integers.

**Q: Why is SHAP slow?**
*   **A:** SHAP calculations are computationally intensive. For the web app, we use a simplified TreeExplainer. Large batch explanations may take time.

---

## 🛣️ Project Roadmap

We are constantly improving the system. Upcoming features include:

*   **Phase 1 (Current)**: Real-time detection with XGBoost and SHAP.
*   **Phase 2**: Integration of **Graph Neural Networks (GNN)** to detect money laundering rings and circular trading.
*   **Phase 3**: **Active Learning Loop** allowing analysts to correct false positives, retraining the model automatically.
*   **Phase 4**: **Docker & Kubernetes** support for scalable cloud deployment.
*   **Phase 5**: **REST API** with FastAPI for integration with external banking backends.

---

## 🤝 Contributing

We welcome contributions from the open-source community!

1.  **Fork** the project.
2.  **Create** your feature branch (`git checkout -b feature/NewFeature`).
3.  **Commit** your changes (`git commit -m 'Add NewFeature'`).
4.  **Push** to the branch (`git push origin feature/NewFeature`).
5.  **Open** a Pull Request.

Please ensure you follow the existing code style (PEP 8) and include comments where necessary.

---

## 📄 License & Acknowledgements

*   **License**: Distributed under the **MIT License**. See `LICENSE` for more information.
*   **Data Source**: Based on the synthetic PaySim dataset.
*   **Tools**: Built with Streamlit, XGBoost, and SHAP.

---
*This project was developed for educational and research purposes. While highly accurate, it is a simulation and should be rigorously tested before production use in financial systems.*
