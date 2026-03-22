import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def attempt_94_raw():
    print("--- EXTREME PURE-RAW EMPIRICAL SQUEEZE (94% Target Search) ---")
    
    # 1. Load Data
    csv_path = "backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(csv_path):
        print("Missing dataset.")
        return
        
    df = pd.read_csv(csv_path)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True, errors='ignore')
    
    # 2. FEATURE ENGINEERING (Extreme Signal Extraction)
    df['MonthlyIncome_Per_Age'] = df['MonthlyIncome'] / (df['Age'] + 1)
    df['YearsSincePromotion_Ratio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['Satisfaction_Index'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction']) / 3
    df['Workload_Index'] = df['NumCompaniesWorked'] * df['TotalWorkingYears']
    df['Tenure_In_Role'] = df['YearsInCurrentRole'] / (df['Age'] + 1)
    
    cat_features = ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']
    num_features = [c for c in df.columns if c not in cat_features + ['Attrition']]
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # 3. 70/30 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # 4. PREPROCESSING
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('scaler', RobustScaler()), ('power', PowerTransformer())]), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # 5. MODEL ARCHITECTURE (High-Capacity Complex Stacking)
    rf = RandomForestClassifier(n_estimators=2000, max_depth=30, class_weight='balanced', random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.005, max_depth=10, scale_pos_weight=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
    gb = GradientBoostingClassifier(n_estimators=2000, learning_rate=0.005, max_depth=6, random_state=42)
    
    stack = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model), ('gb', gb)],
        final_estimator=LogisticRegression(class_weight='balanced', C=100),
        cv=5
    )
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.8)),
        ('classifier', stack)
    ])
    
    print("Training Extreme Stacking Ensemble (Zero Sharpening, 0.5 Threshold)...")
    pipeline.fit(X_train, y_train)
    
    # 6. RAW INFERENCE
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- PURE RAW RESULTS (No Refinement) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}")

if __name__ == "__main__":
    attempt_94_raw()
