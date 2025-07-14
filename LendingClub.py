import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_csv("C:\\Users\\UMANG JAISWAL N\\OneDrive\\Desktop\\SEM - 4\\PA LAB\\lending_club_loan_two.csv\\lending_club_loan_two.csv")

# Feature Selection
selected_features = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'emp_length',
                     'home_ownership', 'annual_inc', 'verification_status', 'purpose', 'dti', 'loan_status']
df = df[selected_features]

# Handling Categorical Variables
encoder = LabelEncoder()
df['term'] = encoder.fit_transform(df['term'])
df['grade'] = encoder.fit_transform(df['grade'])
df['emp_length'] = encoder.fit_transform(df['emp_length'].astype(str))
df['home_ownership'] = encoder.fit_transform(df['home_ownership'])
df['verification_status'] = encoder.fit_transform(df['verification_status'])
df['purpose'] = encoder.fit_transform(df['purpose'])
df['loan_status'] = encoder.fit_transform(df['loan_status'])  # Target Variable

# Splitting Data
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Handling Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost with Hyperparameter Tuning
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, cv=3, scoring='accuracy',
                                   n_jobs=-1, n_iter=20, random_state=42)
random_search.fit(X_train, y_train)
tuned_xgb_model = random_search.best_estimator_
y_pred_xgb = tuned_xgb_model.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Tuned XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report (Tuned XGBoost):")
print(classification_report(y_test, y_pred_xgb))

# Confusion Matrix
plt.figure(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix - Tuned XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Enhanced Risk Prediction
def predict_risk_with_score(new_data):
    new_data = scaler.transform(new_data)
    probabilities = tuned_xgb_model.predict_proba(new_data)[:, 1]
    risk_levels = []
    for prob in probabilities:
        if prob < 0.33:
            risk_levels.append("Low Risk")
        elif prob < 0.66:
            risk_levels.append("Medium Risk")
        else:
            risk_levels.append("High Risk")
    return list(zip(probabilities, risk_levels))

# Example Risk Prediction
sample_data = np.array([[15000, 1, 12.5, 500, 2, 5, 1, 60000, 1, 3, 18]])
results = predict_risk_with_score(sample_data)
for prob, risk in results:
    print(f"Predicted Risk Score: {prob:.2f}, Risk Category: {risk}")

# Risk Analysis on Test Set
test_probabilities = tuned_xgb_model.predict_proba(X_test)[:, 1]
test_risk_levels = pd.cut(test_probabilities, bins=[0, 0.33, 0.66, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"])
test_results_df = pd.DataFrame(X_test, columns=X.columns)
test_results_df['Actual'] = y_test.values
test_results_df['Predicted'] = y_pred_xgb
test_results_df['Risk_Probability'] = test_probabilities
test_results_df['Risk_Level'] = test_risk_levels

# Sample of risk analysis
print("\nSample Risk Analysis:")
print(test_results_df[['Risk_Probability', 'Risk_Level', 'Predicted']].head())

# Risk Distribution Visualization
plt.figure(figsize=(8, 5))
sns.countplot(x=test_results_df['Risk_Level'], palette='Spectral')
plt.title("Distribution of Predicted Risk Levels")
plt.xlabel("Risk Category")
plt.ylabel("Number of Loans")
plt.show()

# Save Risk Report (optional)
test_results_df.to_csv("loan_risk_analysis_results.csv", index=False)
print("Risk analysis report saved as 'loan_risk_analysis_results.csv'")

# Additional Visualizations

# 1. Box Plot for Interest Rate by Loan Status
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['loan_status'], y=df['int_rate'], palette='Set2')
plt.title("Interest Rate Distribution by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Interest Rate")
plt.show()

# 2. Bar Chart of Loan Purpose Counts
plt.figure(figsize=(10, 5))
sns.countplot(x=df['purpose'], palette='coolwarm', order=df['purpose'].value_counts().index)
plt.title("Distribution of Loan Purposes")
plt.xticks(rotation=45)
plt.xlabel("Loan Purpose")
plt.ylabel("Count")
plt.show()

# 3. Stacked Bar Chart for Home Ownership vs Loan Status
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='home_ownership', hue='loan_status', multiple='stack', palette='viridis')
plt.title("Loan Status Distribution Across Home Ownership Types")
plt.xlabel("Home Ownership")
plt.ylabel("Count")
plt.show()

# 4. Heatmap of Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. KDE Plot for Income vs Loan Amount
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='annual_inc', y='loan_amnt', fill=True, cmap="Blues")
plt.title("Kernel Density Estimate: Income vs Loan Amount")
plt.xlabel("Annual Income")
plt.ylabel("Loan Amount")
plt.show()

# Pie Chart for Loan Status Distribution
plt.figure(figsize=(8, 8))
loan_status_counts = df['loan_status'].value_counts()
plt.pie(loan_status_counts, labels=loan_status_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'salmon'])
plt.title("Distribution of Loan Status")
plt.axis('equal')  # Equal aspect ratio ensures that pie chart is a circle.
plt.show()

# Violin Plot for Loan Amount by Grade
plt.figure(figsize=(10, 6))
sns.violinplot(x='grade', y='loan_amnt', data=df, palette='muted')
plt.title("Violin Plot of Loan Amount by Grade")
plt.xlabel("Loan Grade")
plt.ylabel("Loan Amount")
plt.show()

# Feature Importance from Random Forest
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.title("Feature Importances from Random Forest")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

from sklearn.metrics import roc_curve, auc

# Calculate ROC Curve
fpr_log, tpr_log, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:, 1])
roc_auc_log = auc(fpr_log, tpr_log)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, tuned_xgb_model.predict_proba(X_test)[:, 1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, color='blue', label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_log))
plt.plot(fpr_rf, tpr_rf, color='green', label='Random Forest (AUC = {:.2f})'.format(roc_auc_rf))
plt.plot(fpr_xgb, tpr_xgb, color='red', label='XGBoost (AUC = {:.2f})'.format(roc_auc_xgb))
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

print("\nModel Training, Risk Analysis, Prediction, and Visualization Completed Successfully!")
