import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("C:\\Users\\UMANG JAISWAL N\\OneDrive\\Desktop\\SEM - 4\\PA LAB\\lending_club_loan_two.csv\\lending_club_loan_two.csv")

print("Dataset Preview:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values (Before Handling):")
print(df.isnull().sum())

# Handle Missing Values
df.fillna(df.select_dtypes(include=['number']).median(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing Values (After Handling):")
print(df.isnull().sum())

print("- Missing values in numerical columns were handled using median imputation to prevent data distortion.")
print("- Categorical missing values were filled using the most frequent category (mode), ensuring consistency.")
print("- This approach minimizes bias while preserving data integrity for accurate analysis.")

# Remove Duplicates
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate rows removed successfully.")
else:
    print("No duplicate rows found.")

print("\n- Duplicates were identified using the .duplicated() method, which checks for repeated rows.")
print("- If duplicates were present, they were removed using the .drop_duplicates() function.")
print("- Removing duplicates ensures data accuracy and prevents skewed analysis.")

# Outlier Removal using IQR
print("\nOutlier Removal using IQR (Interquartile Range)")

def remove_outliers(df, columns):
    outlier_counts = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts[col] = outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print("Outliers removed using IQR:")
    print(outlier_counts)
    return df

numerical_cols = df.select_dtypes(include=['number']).columns
df = remove_outliers(df, numerical_cols)

print("\n- Outliers were detected and removed using the IQR method.")
print("- Values beyond 1.5 times the IQR from the first and third quartiles were considered outliers.")
print("- This reduces the impact of extreme values and improves model accuracy.")

# Visualization 1: Heatmap of Feature Correlations
df_numeric = df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 6))
sns.heatmap(df_numeric.corr(), cmap="coolwarm", annot=True)
plt.title("Feature Correlation Matrix")
plt.show()

# Visualization 2: Interest Rate Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['int_rate'], bins=30, kde=True, color='blue')
plt.title("Interest Rate Distribution")
plt.xlabel("Interest Rate")
plt.ylabel("Frequency")
plt.show()

# Visualization 3: Loan Amount Distribution by Loan Status
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['loan_status'], y=df['loan_amnt'], palette='coolwarm')
plt.title("Loan Amount Distribution by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")
plt.show()

# Visualization 4: Distribution of Loan Amounts
plt.figure(figsize=(8, 5))
sns.histplot(df['loan_amnt'], bins=30, kde=True, color='green')
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")
plt.show()

# Visualization 5: Count plot for Loan Status
plt.figure(figsize=(8, 5))
sns.countplot(x=df['loan_status'], palette='coolwarm')
plt.title("Loan Status Count")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()

# Visualization 6: Debt-to-Income (DTI) Ratio Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['dti'], bins=30, kde=True, color='purple')
plt.title("Debt-to-Income (DTI) Ratio Distribution")
plt.xlabel("DTI Ratio")
plt.ylabel("Frequency")
plt.show()

# Visualization 7: Relationship between Interest Rate and Loan Amount
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['loan_amnt'], y=df['int_rate'], alpha=0.5)
plt.title("Interest Rate vs Loan Amount")
plt.xlabel("Loan Amount")
plt.ylabel("Interest Rate")
plt.show()

# Visualization 8: Loan Term Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=df['term'], palette='coolwarm')
plt.title("Loan Term Distribution")
plt.xlabel("Term (Months)")
plt.ylabel("Count")
plt.show()

print("\nInference:")
print("- Loan amounts tend to cluster in a specific range, indicating lending preferences.")
print("- Interest rates are spread out, suggesting varying creditworthiness of borrowers.")
print("- Higher loan amounts often correlate with higher interest rates, possibly indicating greater risk.")
print("- Loan status distribution shows a clear distinction between approved and defaulted loans.")
print("- Debt-to-Income (DTI) ratio varies significantly, affecting loan approvals.")
print("- Borrowers with higher loan-to-income ratios may be more likely to default.")
print("- Lenders should consider adjusting interest rates based on risk assessment to minimize defaults.")

print("\nExploratory Data Analysis Completed Successfully!")

# Save the cleaned DataFrame to a specific folder
output_path = "C:\\Users\\UMANG JAISWAL N\\OneDrive\\Desktop\\SEM - 4\\PA LAB\\cleaned_lending_club_data.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned data has been saved to: {output_path}")
