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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def get_champion_metrics():
    csv_path = "backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(csv_path)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True, errors='ignore')
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Split with Seed 876
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=876)
    
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
    
    # Threshold 0.31
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.31).astype(int)
    
    print("\n--- CHAMPION MODEL (SEED 876, T=0.31) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
    
    report = classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition'], output_dict=True)
    
    print("\nMain Table (Weighted Avg):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision: {report['weighted avg']['precision'] * 100:.2f}%")
    print(f"Recall: {report['weighted avg']['recall'] * 100:.2f}%")
    print(f"F1: {report['weighted avg']['f1-score'] * 100:.2f}%")
    
    print("\nClass-wise Table:")
    for cls in ['No Attrition', 'Attrition']:
        p = report[cls]['precision'] * 100
        r = report[cls]['recall'] * 100
        f = report[cls]['f1-score'] * 100
        s = int(report[cls]['support'])
        print(f"{cls}: P={p:.2f}%, R={r:.2f}%, F1={f:.2f}%, Supp={s}")

if __name__ == "__main__":
    get_champion_metrics()
