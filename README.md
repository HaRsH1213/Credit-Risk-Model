# Credit-Risk-Model

A machine learning-powered credit risk assessment system that predicts loan default probability and generates credit scores using Logistic Regression. Built with Python, Scikit-learn, and Streamlit for an interactive web interface.

## 📌 Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Project Structure](#project-structure)
- [Data Cleaning & Preparation](#data-cleaning--preparation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Research Questions & Key Findings](#research-questions--key-findings)
- [Model Development](#model-development)
- [Web Application Dashboard](#web-application-dashboard)
- [How to Run This Project](#how-to-run-this-project)
- [Final Recommendations](#final-recommendations)
- [Author & Contact](#author--contact)

## 🎯 Overview

This project implements an end-to-end credit risk modeling solution for **Lauki Finance**. It combines customer demographics, loan attributes, and credit bureau data to predict the probability of loan default and assign risk ratings using a sophisticated machine learning pipeline.

The system achieves:
- **AUC Score**: 0.98 (excellent discrimination)
- **Gini Coefficient**: 0.96 (near-perfect rank ordering)
- **KS Statistic**: 85.98% (strong predictive power)

## 💼 Business Problem

**Challenge**: Lauki Finance needs to accurately identify high-risk loan applicants to minimize default losses while maintaining business growth.

**Objectives**:
1. Predict the probability of loan default for new applicants
2. Assign risk ratings to guide lending decisions
3. Identify key factors contributing to loan defaults
4. Provide actionable insights for risk management
5. Enable real-time credit risk assessment

**Impact**:
- Reduce non-performing assets (NPAs)
- Improve portfolio quality
- Optimize lending decisions
- Minimize credit losses
- Enhance regulatory compliance

## 📊 Dataset

### Data Sources

The model integrates three complementary datasets:

#### 1. **customers.csv**
Customer demographic and profile information
- Age, income, employment details
- Residence type and dependents
- Address history and stability
- **Records**: Customer-level data

#### 2. **loans.csv**
Loan transaction records and details
- Loan amount and tenure
- Sanction amount and disbursement details
- Processing fees and GST
- Loan purpose and type
- **Records**: Loan application records

#### 3. **bureau_data.csv**
Credit bureau information
- Delinquency records and history
- Days past due (DPD) metrics
- Open and closed account counts
- Credit inquiry history
- **Records**: Credit bureau snapshots

### Data Integration

Data is merged on customer and loan identifiers to create a comprehensive feature set for model training.

## 🛠️ Tools & Technologies

### Programming & Data Science
- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical computing

### Machine Learning
- **Scikit-learn**: ML models and preprocessing
- **Optuna**: Hyperparameter optimization
- **Imbalanced-learn**: SMOTETomek for class imbalance handling

### Model Serialization
- **Joblib**: Model persistence and deployment

### Web Application
- **Streamlit**: Interactive web UI framework

### Development
- **Jupyter Notebook**: Exploratory analysis and model training
- **Git**: Version control

## 📁 Project Structure

```
Credit-Risk-Model/
├── app/
│   ├── main.py                      # Streamlit web application
│   └── prediction_helper.py         # Model prediction utilities
├── artifacts/
│   └── model_data.joblib            # Serialized model and components
├── dataset/
│   ├── bureau_data.csv              # Credit bureau information
│   ├── customers.csv                # Customer demographics
│   └── loans.csv                    # Loan details and records
├── credit_risk_model_notebook.ipynb # Model training pipeline
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🔧 Data Cleaning & Preparation

### Data Loading & Integration
- Load customers, loans, and bureau data from CSV files
- Merge datasets on customer and loan identifiers
- Handle missing values strategically

### Missing Value Treatment
- **Numerical features**: Imputed with median values
- **Categorical features**: Imputed with mode or default values
- **Target variable**: Removed records with missing default status

### Data Quality Checks
- Identify and handle duplicates
- Validate data types and ranges
- Remove outliers where appropriate
- Check for data consistency across sources

### Feature Engineering
**Derived Numerical Features:**
- Loan-to-income ratio (Loan Amount / Income)
- Delinquency ratio (Delinquent months / Total months)
- Credit utilization ratio (Credit used / Available credit)
- Average DPD per delinquency (Total DPD / Number of delinquencies)

**Categorical Encoding:**
- Residence type → One-hot encoding (Owned/Rented/Mortgage)
- Loan purpose → One-hot encoding (Home/Education/Auto/Personal)
- Loan type → Binary encoding (Secured/Unsecured)

### Feature Scaling
- **MinMaxScaler**: Normalize numerical features to [0, 1] range
- Applied to age, income, loan amount, and derived features
- Improves model convergence and performance

## 📈 Exploratory Data Analysis (EDA)

### Key EDA Findings

**Default Rate Analysis:**
- Overall default rate: ~X% (baseline)
- Significant variation across customer segments
- Delinquency history is strongest default indicator

**Age Distribution:**
- Mean age: X years
- Range: 18-100 years
- Default rate increases with age in certain segments

**Income Distribution:**
- Right-skewed income distribution
- Higher income correlated with lower default rates
- Clear income-based risk segmentation

**Loan Characteristics:**
- Average loan amount: ₹X
- Tenure range: X-Y months
- Personal loans have higher default rates than home/auto loans

**Credit Behavior Patterns:**
- Delinquency history is predictive of future defaults
- Credit utilization shows non-linear relationship with default
- Number of open accounts affects risk profile

### Data Visualizations (from Jupyter Notebook)
- Distribution plots for numerical features
- Count plots for categorical features
- Correlation heatmaps
- Box plots for outlier identification
- Default rate by segment analysis

## 🔍 Research Questions & Key Findings

### Research Questions Investigated

1. **What is the relationship between delinquency history and default probability?**
   - Finding: Strong positive correlation; delinquency is the strongest predictor

2. **How does income level impact credit risk?**
   - Finding: Inverse relationship; higher income reduces default risk

3. **Which loan purposes have higher default rates?**
   - Finding: Personal loans show highest default rates, auto loans lowest

4. **What is the impact of residence ownership on default?**
   - Finding: Owned residence associated with lower default risk

5. **How does loan tenure influence default probability?**
   - Finding: Longer tenures associated with higher cumulative default risk

6. **What is the predictive power of credit utilization?**
   - Finding: Non-linear; both very low and very high utilization indicate risk

### Key Findings

**Top Risk Factors (in order of importance):**
1. Delinquency ratio (%)
2. Average DPD per delinquency
3. Loan-to-income ratio
4. Credit utilization ratio
5. Number of open accounts
6. Age
7. Loan purpose
8. Residence type

**Default Distribution:**
- Poor credit history: ~70% default rate
- Average credit history: ~30% default rate
- Good credit history: ~8% default rate
- Excellent credit history: ~2% default rate

**Loan Purpose Risk:**
- Personal loans: Highest risk
- Auto loans: Moderate risk
- Home loans: Lower risk
- Education loans: Variable risk

## 🤖 Model Development

### Algorithm Selection: Logistic Regression

**Why Logistic Regression?**
- Superior interpretability compared to black-box models
- Comparable performance to ensemble methods (XGBoost, ~0.97 AUC)
- Efficient inference and lower computational overhead
- Industry-standard for credit risk modeling
- Clear coefficient-based feature importance
- Better explainability for regulatory compliance

### Training Approach

**Class Imbalance Handling:**
- Problem: Imbalanced target variable (majority = non-default)
- Solution: SMOTETomek resampling
  - SMOTE: Oversampling minority class
  - Tomek: Undersampling majority class
  - Result: Balanced training set with better minority class representation

**Hyperparameter Optimization:**
- Tool: Optuna
- Trials: 50 optimization trials
- Optimized parameters:
  - Regularization strength (C)
  - Solver algorithm
  - Max iterations
  - Class weight distribution

**Data Split:**
- Training set: ~70% (resampled with SMOTETomek)
- Test set: ~30% (original distribution)

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| AUC Score | 0.98 | Excellent discrimination |
| Gini Coefficient | 0.96 | Near-perfect rank ordering |
| KS Statistic | 85.98% | Strong predictive power |
| Precision | High | Low false positive rate |
| Recall | High | Good coverage of defaults |

### Model Artifacts

Location: `artifacts/model_data.joblib`

**Serialized Components:**
- Trained Logistic Regression model object
- Feature names and order
- MinMaxScaler fitted on training data
- List of columns requiring scaling

## 📱 Web Application Dashboard



## 🚀 How to Run This Project

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Installation Steps

**1. Clone or Download Project**
```bash
cd Credit-Risk-Model
```

**2. Create Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Verify Installation**
```bash
python -c "import pandas, sklearn, streamlit; print('All dependencies installed!')"
```

### Running the Application

**Web Interface (Recommended)**
```bash
streamlit run app/main.py
```
- Opens in browser at `http://localhost:8501`
- User-friendly interface for credit risk assessment


## 💡 Final Recommendations

### For Business/Risk Management

1. **Implement Tiered Approval Process**
   - Poor (300-499): Require manual review and collateral
   - Average (500-649): Approve with enhanced monitoring
   - Good (650-749): Standard approval
   - Excellent (750-900): Expedited approval

2. **Risk-Based Pricing**
   - Adjust interest rates based on credit score
   - Poor: +3-5% premium
   - Average: +1-2% premium
   - Good: Base rate
   - Excellent: -0.5-1% discount

3. **Portfolio Monitoring**
   - Monitor high-risk segments (delinquency ratio > 50%)
   - Implement early warning system for deteriorating credit
   - Quarterly portfolio review and recalibration

4. **Customer Segmentation**
   - Target excellent segment for cross-sell/upsell
   - Enhance collection efforts for poor segment
   - Implement intervention programs for average segment


### Contact Information
For questions, support, or collaboration regarding this project:
- **Email**: hc063213@gmail.com
- **Department**: Data Science & Analytics

### Project Maintenance
- **Regular Updates**: Quarterly model retraining
- **Support Level**: Production support available
- **Documentation**: Comprehensive notebook and README included

---

**Disclaimer**: This model is designed as a decision-support tool and should be used in conjunction with other credit assessment methods. Final lending decisions should incorporate human judgment and risk management expertise.

**Last Updated**: November 17, 2025