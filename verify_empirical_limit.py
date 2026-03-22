import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

# LOAD DATA
csv_path = "backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(csv_path)
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

# FEATURE ENGINEERING (Ratios & Interactions)
df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1)
df['TenurePerAge'] = df['YearsAtCompany'] / (df['Age'] + 1)
df['JobIntensity'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)

cat_features = ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']
num_features = [c for c in df.columns if c not in cat_features + ['Attrition']]

X = df.drop('Attrition', axis=1)
y = df['Attrition']

# 70/30 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# PIPELINE
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# MODELS (Targeted at maximum natural high-accuracy)
# We use Stacking to allow a meta-learner to find the best weighted combination
estimators = [
    ('rf', RandomForestClassifier(n_estimators=1000, max_depth=20, class_weight='balanced', random_state=42)),
    ('svc', SVC(probability=True, class_weight='balanced', C=10, gamma='scale', random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, scale_pos_weight=5, eval_metric='logloss', random_state=42, use_label_encoder=False))
]

stack = StackingClassifier(
    estimators=estimators, 
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=5
)

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stack)
])

print("Executing Extreme Empirical Training (No Boost, No Sharpening)...")
full_pipeline.fit(X_train, y_train)

# EVALUATION (Standard 0.5 Threshold)
y_pred = full_pipeline.predict(X_test)
y_probs = full_pipeline.predict_proba(X_test)[:, 1]

print("\n--- FINAL RAW RESULTS (Standard 0.5 Threshold) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
