# 🏦 Financial Risk Modeling - Automated Loan Approval System

An end-to-end machine learning solution for automated loan approval decisions and risk assessment. This project demonstrates advanced feature engineering, model optimization, and explainable AI techniques for financial applications.

## 🎯 Project Overview

This project addresses two critical challenges in automated lending:

1. **Loan Approval Classification**: Predict whether a loan application should be approved or denied with 99.75% precision
2. **Risk Score Prediction**: Accurately estimate applicant risk scores to determine appropriate interest rates (R² = 0.9998)

The solution processes 20,000 loan applications, engineers 13 new financial features, and implements SHAP-based explainability to ensure stakeholder trust and regulatory compliance.

## ✨ Key Features

- **🔍 Comprehensive Data Analysis**: Exploratory data analysis with visualization of key patterns and correlations
- **⚙️ Advanced Feature Engineering**: Created 13 domain-specific features including financial ratios (DTI, LTV, Liquidity)
- **🤖 Multiple ML Algorithms**: Tested Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
- **📊 Class Imbalance Handling**: Implemented SMOTE for balanced training
- **🎛️ Hyperparameter Optimization**: Grid and Randomized Search for optimal model tuning
- **💡 Model Explainability**: SHAP analysis for transparent decision-making
- **📈 Performance Visualization**: ROC curves, confusion matrices, residual plots, and feature importance charts
- **💾 Production-Ready Models**: Serialized models and preprocessors for deployment

## 📊 Dataset

**Size**: 20,000 loan applications  
**Features**: 34 original features + 13 engineered features  
**Targets**: 
- `LoanApproved` (Binary: 0=Denied, 1=Approved)
- `RiskScore` (Continuous: Risk assessment score)

### Key Features Include:
- **Personal Information**: Age, Employment Status, Education Level, Experience
- **Financial Metrics**: Annual Income, Credit Score, Total Assets, Total Liabilities
- **Credit Behavior**: Payment History, Credit Utilization, Number of Credit Inquiries
- **Loan Details**: Loan Amount, Duration, Purpose, Interest Rate
- **Risk Indicators**: Bankruptcy History, Previous Defaults, Debt Payments

### Required Libraries:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
shap>=0.41.0
matplotlib>=3.4.0
seaborn>=0.11.0
openpyxl>=3.0.0
```

## 💻 Usage

### Option 1: Run Jupyter Notebook (Recommended)
```bash
jupyter notebook financial_risk_modeling.ipynb
```

### Option 2: Run Python Script
```bash
python financial_risk_modeling.py
```

### Option 3: Use Pre-trained Models
```python
import pickle
import pandas as pd

# Load the classification model
with open('final_clf_model.pkl', 'rb') as f:
    clf_model = pickle.load(f)

# Load the regression model
with open('final_reg_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

# Load scalers
with open('scaler_clf.pkl', 'rb') as f:
    scaler_clf = pickle.load(f)

# Make predictions
# X_new = your new data (preprocessed)
loan_approval = clf_model.predict(X_new)
risk_score = reg_model.predict(X_new)
```

## 🏆 Model Performance

### Classification Model (Loan Approval)
**Algorithm**: XGBoost Classifier (Tuned)

| Metric | Score |
|--------|-------|
| **Precision** | 99.75% |
| **Recall** | 99.80% |
| **F1-Score** | 99.77% |
| **ROC-AUC** | 99.99% |
| **Accuracy** | 99.75% |

### Regression Model (Risk Score)
**Algorithm**: Gradient Boosting Regressor (Tuned)

| Metric | Score |
|--------|-------|
| **RMSE** | 0.7041 |
| **MAE** | 0.4999 |
| **R² Score** | 0.9998 |

## 🔬 Methodology

### 1. Data Preprocessing
- ✅ Handled missing values and duplicates
- ✅ Detected and capped outliers using IQR method (3x threshold)
- ✅ Validated domain constraints (credit scores, utilization rates)
- ✅ Converted date features to datetime format

### 2. Feature Engineering
Created 13 new features:

**Financial Ratios**:
- Debt-to-Income (DTI) Ratio
- Loan-to-Value (LTV) Ratio
- Liquidity Ratio
- Asset Coverage Ratio
- Income-to-Loan Ratio

**Derived Metrics**:
- Monthly Income
- Disposable Income
- Net Worth
- Credit Quality Score (composite)
- High Credit Utilization Flag

**Temporal Features**:
- Application Year, Month, Day of Week

**Categorical Binning**:
- Age Groups (Young, Middle Age, Mature, Senior)
