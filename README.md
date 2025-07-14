# 💰 Lending Club Loan Risk Analysis using Machine Learning

This project performs detailed analysis, prediction, and risk assessment of Lending Club loan applications using advanced machine learning techniques like **Logistic Regression**, **Random Forest**, and **XGBoost** with hyperparameter tuning and SMOTE for class imbalance.

---

## 🎯 Objective

To build a robust ML pipeline that:
- Predicts loan approval outcomes
- Evaluates credit risk
- Categorizes loans as **Low**, **Medium**, or **High Risk**
- Visualizes loan trends and feature importance

---

## 📂 Dataset

- 📄 **File**: `lending_club_loan_two.csv`
- 📌 **Columns used**:
  - `loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `emp_length`
  - `home_ownership`, `annual_inc`, `verification_status`, `purpose`, `dti`
  - `loan_status` (target)

---

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **Pandas**, **NumPy** for data manipulation
- **Matplotlib**, **Seaborn** for visualization
- **Scikit-learn** for ML models
- **XGBoost** for tuned gradient boosting
- **Imbalanced-learn (SMOTE)** for handling imbalance
- **LabelEncoder**, **StandardScaler**, **RandomizedSearchCV**

---

## 📊 Features Implemented

✅ Preprocessing  
✅ Label encoding of categorical variables  
✅ SMOTE for class balancing  
✅ Feature scaling  
✅ Train-test split  
✅ Models: Logistic Regression, Random Forest, Tuned XGBoost  
✅ Classification reports + confusion matrix  
✅ ROC-AUC curves  
✅ Risk scoring system (Low, Medium, High)  
✅ Risk visualization  
✅ Correlation analysis  
✅ Export to CSV

---

## 🔍 Model Evaluation

Each model is evaluated using:
- **Accuracy**
- **Classification report**
- **Confusion matrix**
- **ROC-AUC curve**

XGBoost undergoes **hyperparameter tuning** using `RandomizedSearchCV`.

---

## 📉 Visualizations Included

- 📌 Confusion Matrix (XGBoost)
- 📌 Risk Distribution Bar Chart
- 📌 Box Plot: Interest Rate vs Loan Status
- 📌 Count Plot: Loan Purpose
- 📌 Stacked Histogram: Home Ownership vs Loan Status
- 📌 Correlation Heatmap
- 📌 KDE Plot: Annual Income vs Loan Amount
- 📌 Pie Chart: Loan Status
- 📌 Violin Plot: Loan Amount by Grade
- 📌 Feature Importance (Random Forest)
- 📌 ROC Curves for all models

---

## 🧪 Risk Scoring System

Uses `predict_proba()` from XGBoost to assign:
- **Low Risk** (0.0 – 0.33)
- **Medium Risk** (0.34 – 0.66)
- **High Risk** (0.67 – 1.0)

```python
sample_data = np.array([[15000, 1, 12.5, 500, 2, 5, 1, 60000, 1, 3, 18]])
results = predict_risk_with_score(sample_data)
