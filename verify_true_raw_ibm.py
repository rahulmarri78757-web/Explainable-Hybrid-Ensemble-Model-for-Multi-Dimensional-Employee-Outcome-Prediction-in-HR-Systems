import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

def verify_true_raw():
    print("\n--- TRUE RAW PERFORMANCE VERIFICATION (IBM HR Dataset) ---")
    print("Conditions: No Calibration, No Scaling, No Sharpening, Standard 0.5 Threshold")
    
    # 1. Load Data
    csv_path = "backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Target Encoding
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Drop Useless Columns
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True, errors='ignore')
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # 2. Confirm Dataset Sizes
    total_samples = len(df)
    print(f"\n[Dataset Info]")
    print(f"Total Dataset Size: {total_samples}")
    
    # 3. Stratified 70/30 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    print(f"Training Set Size: {len(X_train)}")
    print(f"Test Set Size: {len(X_test)}")
    
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()
    print(f"Test Class Distribution: Class 0 (No): {test_counts.get(0, 0)}, Class 1 (Yes): {test_counts.get(1, 0)}")
    
    # 4. Pipeline Construction
    cat_features = ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']
    num_features = [c for c in X.columns if c not in cat_features]
    
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    
    # 5. Model Architecture (Weighted for Imbalance)
    # Note: explicit random_state for reproducibility
    rf = RandomForestClassifier(n_estimators=1000, max_depth=20, class_weight='balanced', random_state=42)
    lr = LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42)
    svc = SVC(probability=True, class_weight='balanced', C=10, random_state=42)
    
    # Calculate scale_pos_weight for XGBoost
    neg_count = train_counts.get(0, 0)
    pos_count = train_counts.get(1, 1)
    scale_pos_weight = neg_count / pos_count
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.01, 
        max_depth=6,
        scale_pos_weight=scale_pos_weight, 
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('svc', svc), ('xgb', xgb_model)],
        voting='soft'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])
    
    # 6. Training
    print("\nTraining Ensemble (Pipeline)...")
    pipeline.fit(X_train, y_train)
    
    # 7. Raw Inference
    print("Predicting on Test Set...")
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    # 8. Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_probs)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n--- HOLD-OUT TEST RESULTS (RAW) ---")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC:   {roc:.4f}")
    
    print("\n[Confusion Matrix]")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    
    # 9. 5-Fold Stratified Cross-Validation
    print("\nRunning 5-Fold Stratified Cross-Validation (Full Pipeline)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_f1_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
    
    print(f"Mean F1: {np.mean(train_f1_scores):.4f}")
    print(f"Std Dev: {np.std(train_f1_scores):.4f}")

if __name__ == "__main__":
    verify_true_raw()
