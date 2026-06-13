import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# New ML Algorithms
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

import warnings
from urllib.parse import quote_plus  # Fixes the '@' in password issue

# Suppress minor warnings for a cleaner console output
warnings.filterwarnings('ignore')

import joblib 
from fastapi import FastAPI
# ==========================================
# CONFIGURATION
# ==========================================
db_user = 'root'                 
db_password = 'Charan@123'       # Your MySQL password
db_host = 'localhost'            
db_name = 'retail_sales_db'      
file_path = 'Retail_Transaction_Dataset.csv' 

# ==========================================
# PHASE 1: DATA COLLECTION & DB SETUP
# ==========================================
print("Phase 1: Connecting to MySQL and loading raw data...")

# URL-encode the password safely
encoded_password = quote_plus(db_password)
engine = create_engine(f'mysql+pymysql://{db_user}:{encoded_password}@{db_host}:3306/{db_name}')

df_raw = pd.read_csv(file_path)

# Clean column names for SQL compatibility
df_raw.rename(columns={'DiscountApplied(%)': 'DiscountApplied_Percent'}, inplace=True)

# Push raw data to MySQL
df_raw.to_sql(name='transactions', con=engine, if_exists='replace', index=False)
print(" -> Data successfully loaded to MySQL.\n")

# ==========================================
# PHASE 2: DATA PREPARATION & ML PIPELINE
# ==========================================
print("Phase 2: Preparing Data and Running ML Algorithms...")
df = df_raw.copy()
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# ---------------------------------------------------------
# Feature Engineering for Machine Learning
# Selecting relevant columns and converting text to numeric
# ---------------------------------------------------------
df_ml = df[['Quantity', 'Price', 'DiscountApplied_Percent', 'ProductCategory', 'PaymentMethod', 'TotalAmount']].copy()

# One-Hot Encoding: Converts text categories into binary columns (0s and 1s)
df_ml = pd.get_dummies(df_ml, columns=['ProductCategory', 'PaymentMethod'], drop_first=True)

# ---------------------------------------------------------
# ML Task 1: REGRESSION (Predicting TotalAmount)
# Algorithm: Gradient Boosting Regressor
# Metrics: MSE, R2 Score
# ---------------------------------------------------------
print(" -> Running Algorithm 1: Gradient Boosting Regressor...")
X_reg = df_ml.drop('TotalAmount', axis=1)
y_reg = df_ml['TotalAmount']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"    * Mean Squared Error (MSE): {mse:.2f}")
print(f"    * R-squared (R2): {r2:.4f}\n")

# ---------------------------------------------------------
# ML Task 2: CLASSIFICATION (Predicting High vs Low Value Transaction)
# Algorithm: Random Forest Classifier
# Metrics: Accuracy, Confusion Matrix
# ---------------------------------------------------------
print(" -> Running Algorithm 2: Random Forest Classifier...")
# Create Target: 1 if TotalAmount > Median (High Value), 0 if below (Low Value)
median_amount = df['TotalAmount'].median()
df_ml['High_Value_Class'] = (df_ml['TotalAmount'] > median_amount).astype(int)

X_clf = df_ml.drop(['TotalAmount', 'High_Value_Class'], axis=1)
y_clf = df_ml['High_Value_Class']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)
y_pred_clf = clf_model.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
conf_matrix = confusion_matrix(y_test_clf, y_pred_clf)

print(f"    * Accuracy: {accuracy * 100:.2f}%")
print(f"    * Confusion Matrix:\n{conf_matrix}\n")

# ==========================================
# PHASE 3: VISUALIZATIONS & EXPORT
# ==========================================
print("Phase 3: Generating Visualizations...")

# Create a 2x2 grid for charts
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Revenue by Product Category
category_sales = df.groupby('ProductCategory')['TotalAmount'].sum().reset_index().sort_values(by='TotalAmount', ascending=False)
sns.barplot(data=category_sales, x='TotalAmount', y='ProductCategory', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Revenue by Product Category', fontsize=12)
axes[0, 0].set_xlabel('Total Sales ($)')

# Plot 2: Payment Method Popularity
payment_counts = df['PaymentMethod'].value_counts().reset_index()
sns.barplot(data=payment_counts, x='count', y='PaymentMethod', ax=axes[0, 1], palette='magma')
axes[0, 1].set_title('Transactions by Payment Method', fontsize=12)
axes[0, 1].set_xlabel('Number of Transactions')

# Plot 3: ML Regression - Actual vs Predicted
sns.scatterplot(x=y_test_reg, y=y_pred_reg, ax=axes[1, 0], alpha=0.5, color='green')
axes[1, 0].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[1, 0].set_title('Regression: Actual vs Predicted TotalAmount', fontsize=12)
axes[1, 0].set_xlabel('Actual TotalAmount')
axes[1, 0].set_ylabel('Predicted TotalAmount')

# Plot 4: ML Classification - Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], cbar=False)
axes[1, 1].set_title('Classification: Confusion Matrix (High Value)', fontsize=12)
axes[1, 1].set_xlabel('Predicted Label')
axes[1, 1].set_ylabel('True Label')

plt.tight_layout()
plt.savefig('Sales_Dashboard_With_ML.png')
print(" -> Visuals saved to 'Sales_Dashboard_With_ML.png'.")

# Export to Excel
excel_filename = 'Retail_Sales_Master_Report.xlsx'
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
    
    # Save the ML predictions/metrics back to the excel file for review
    metrics_df = pd.DataFrame({
        'Metric': ['MSE', 'R2 Score', 'Accuracy'],
        'Value': [mse, r2, accuracy]
    })
    metrics_df.to_excel(writer, sheet_name='ML_Metrics', index=False)

print(f" -> Data successfully exported to '{excel_filename}'.")
print("\n--- Process Complete ---")

# Save the REGRESSION model
joblib.dump(reg_model, "reg_model.joblib")

# Save the EXACT columns used during training
model_columns = X_train_reg.columns.tolist()
joblib.dump(model_columns, "model_columns.joblib")

print("Models saved properly for FastAPI!")