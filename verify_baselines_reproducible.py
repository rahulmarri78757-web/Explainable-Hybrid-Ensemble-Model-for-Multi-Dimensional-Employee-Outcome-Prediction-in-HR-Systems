import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import time

def verify_baselines_reproducible():
    print("--- Reproducible Baseline Verification ---")
    
    # Load Data
    df = pd.read_csv("backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True, errors='ignore')
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Preprocessing
    num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, [c for c in X.columns if c not in ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']]),
        ('cat', cat_pipeline, ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField'])
    ])
    
    # Models to Compare (Both Default and Balanced)
    models = [
        ("LogReg (Default)", LogisticRegression(max_iter=3000, random_state=42)),
        ("LogReg (Balanced)", LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)),
        ("RF (Default)", RandomForestClassifier(n_estimators=1000, random_state=42)),
        ("RF (Balanced)", RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42)),
        ("SVM (Default)", SVC(probability=True, random_state=42)),
        ("SVM (Balanced)", SVC(probability=True, class_weight='balanced', random_state=42))
    ]
    
    print(f"{'Model':<20} | {'Acc':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 55)
    
    for name, model in models:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"{name:<20} | {acc:.2%}  | {rec:.2%}  | {f1:.2%}")

if __name__ == "__main__":
    verify_baselines_reproducible()
