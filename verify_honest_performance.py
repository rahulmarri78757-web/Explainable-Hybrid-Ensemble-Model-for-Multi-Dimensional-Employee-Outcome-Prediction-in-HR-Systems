import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV

def verify_honest_performance():
    print("\n--- HONEST SCIENTIFIC OPTIMIZATION (Standard ML Techniques) ---")
    print("Techniques: Feature Engineering + SMOTEENN + Isotonic Calibration\n")
    
    # 1. Load Data
    csv_path = "backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(csv_path):
        print("Dataset not found.")
        return

    df = pd.read_csv(csv_path)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True, errors='ignore')
    
    # 2. ADVANCED FEATURE ENGINEERING (Legitimate Signal Creation)
    # Income Ratios
    df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1)
    df['IncomePerEdu'] = df['MonthlyIncome'] / (df['Education'] + 1)
    
    # Stability Ratios
    df['TenureRatio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['LoyaltyIndex'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    df['StabilityMetric'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
    
    # Satisfaction Composite
    df['SatisfactionScore'] = (df['EnvironmentSatisfaction'] + df['JobSatisfaction'] + df['RelationshipSatisfaction']) / 3
    df['DissatisfactionFlag'] = ((df['JobSatisfaction'] < 2) | (df['EnvironmentSatisfaction'] < 2)).astype(int)
    
    # Work Intensity
    df['WorkIntensity'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1) 
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # 3. Stratified 70/30 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # 4. Pipeline Construction
    cat_features = ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']
    num_features = [c for c in X.columns if c not in cat_features]
    
    # Robust scaling handles outliers better than Standard
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()), 
        ('power', PowerTransformer()) # Normalizes skewed distributions
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    
    # 5. Model Architecture (Tuned for Honest Performance)
    # SMOTEENN: Cleans noise while oversampling (Critical for 90%+)
    # XGBoost: Low learning rate for better generalization
    
    rf = RandomForestClassifier(n_estimators=2000, max_depth=None, min_samples_split=5, class_weight='balanced_subsample', random_state=42)
    lr = LogisticRegression(max_iter=5000, class_weight='balanced', C=0.5, random_state=42)
    svc = SVC(probability=True, class_weight='balanced', C=10, gamma='scale', random_state=42)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000, 
        learning_rate=0.005, 
        max_depth=7,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=5, 
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr), ('svc', svc), ('xgb', xgb_model)],
        voting='soft'
    )
    
    # ISOTONIC CALIBRATION (Standard Library Method)
    # This aligns probabilities without "fake" sharpening
    calibrated_ensemble = CalibratedClassifierCV(ensemble, method='isotonic', cv=5)
    
    # Imbalanced Pipeline
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('classifier', calibrated_ensemble)
    ])
    
    # 6. Training & Evaluation
    print("Training Full Pipeline (SMOTEENN + Isotonic Calibration)...")
    pipeline.fit(X_train, y_train)
    
    print("Predicting on Test Set...")
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_probs)
    
    print("\n--- HONEST IBM RESULTS ---")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix: TN:{cm[0,0]}, FP:{cm[0,1]}, FN:{cm[1,0]}, TP:{cm[1,1]}")

if __name__ == "__main__":
    verify_honest_performance()
