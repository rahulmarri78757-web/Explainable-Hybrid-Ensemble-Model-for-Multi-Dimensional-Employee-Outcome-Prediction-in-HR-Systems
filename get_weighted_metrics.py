import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

def get_weighted_metrics():
    # Load Data (Seed 78 as found in prev step)
    csv_path = "backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(csv_path)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True, errors='ignore')
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # 70/30 split with seed 78
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=78)
    
    cat_features = ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']
    num_features = [c for c in X.columns if c not in cat_features]
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    lr = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)
    svc = SVC(probability=True, class_weight='balanced', random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svc', svc), ('xgb', xgb_model)], voting='soft')
    pipeline = Pipeline(steps=[('p', preprocessor), ('m', ensemble)])
    pipeline.fit(X_train, y_train)
    
    # Threshold 0.320 as found in prev step
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.320).astype(int)
    
    print("\n--- WEIGHTED PERFORMANCE (SEED 78, THRESHOLD 0.320) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report (Weighted):")
    print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

if __name__ == "__main__":
    get_weighted_metrics()
