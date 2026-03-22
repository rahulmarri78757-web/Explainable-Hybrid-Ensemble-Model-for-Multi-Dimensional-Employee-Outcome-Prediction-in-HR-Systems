import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

def try_extreme_raw_optimization():
    print("--- Starting Extreme Pure-Raw Optimization (Zero Calibration) ---")
    
    # 1. Load Data
    csv_path = os.path.join(os.path.dirname(__file__), "backend", "app", "ml", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = pd.read_csv(csv_path)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Drop constant/useless features
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True, errors='ignore')
    
    # ADVANCED FEATURE ENGINEERING (Creating more signal for "Raw" performance)
    df['IncomePerLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
    df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['SatisfactionSum'] = df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['RelationshipSatisfaction']
    df['StabilityIndex'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    
    cat_features = ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']
    num_features = [c for c in df.columns if c not in cat_features + ['Attrition']]
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # 2. 70/30 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # 3. Robust Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # 4. Extreme High-Capacity Stacking Ensemble
    # We use very high performance settings to squeeze every bit of signal
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=1500, max_depth=25, min_samples_split=2, class_weight='balanced', random_state=42)),
        ('svc', SVC(probability=True, class_weight='balanced', C=20, gamma='auto', random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=1500, learning_rate=0.005, max_depth=8, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5, random_state=42, use_label_encoder=False, eval_metric='logloss'))
    ]
    
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stack)
    ])
    
    print("Training Extreme Stacking Ensemble (No Calibration, No Sharpening)...")
    pipeline.fit(X_train, y_train)
    
    # 5. PURE RAW INFERENCE (Predict only, 0.5 Threshold)
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- PURE RAW RESULTS (0.5 Threshold) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (Raw Distribution 16%):")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # 6. Check Balanced Raw Limit
    # If the test set was balanced, what would the RAW model do?
    from sklearn.utils import resample
    X_t_bal = X_test.copy()
    X_t_bal['target'] = y_test
    df_no = X_t_bal[X_t_bal.target == 0]
    df_yes = X_t_bal[X_t_bal.target == 1]
    df_yes_up = resample(df_yes, replace=True, n_samples=len(df_no), random_state=42)
    df_bal = pd.concat([df_no, df_yes_up])
    X_eval = df_bal.drop('target', axis=1)
    y_eval = df_bal.target
    
    y_bal_pred = pipeline.predict(X_eval)
    print("\n--- BALANCED RAW LIMIT (Same Model, No Calibration) ---")
    print(f"Balanced Accuracy: {accuracy_score(y_eval, y_bal_pred):.4f}")
    print(f"Balanced F1-Score: {f1_score(y_eval, y_bal_pred):.4f}")

if __name__ == "__main__":
    try_extreme_raw_optimization()
