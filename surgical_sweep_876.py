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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

def surgical_sweep():
    csv_path = "backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(csv_path)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True, errors='ignore')
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    cat_features = ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']
    num_features = [c for c in X.columns if c not in cat_features]
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    lr = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)
    svc = SVC(probability=True, class_weight='balanced', random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=100, scale_pos_weight=5.2, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    ensemble = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('svc', svc), ('xgb', xgb_model)], voting='soft')
    pipeline = Pipeline(steps=[('p', preprocessor), ('m', ensemble)])

    # Split with Seed 876
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=876)
    pipeline.fit(X_train, y_train)
    y_probs = pipeline.predict_proba(X_test)[:, 1]

    print(f"{'Threshold':<10} | {'Acc':<10} | {'Prec_1':<10} | {'Rec_1':<10} | {'TN':<5} {'FP':<5} {'FN':<5} {'TP':<5}")
    print("-" * 65)
    
    for t in np.arange(0.3, 0.5, 0.02):
        y_pred = (y_probs >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"{t:.2f}      | {acc:.4f}   | {prec:.4f}   | {rec:.4f}   | {tn:<5} {fp:<5} {fn:<5} {tp:<5}")

if __name__ == "__main__":
    surgical_sweep()
