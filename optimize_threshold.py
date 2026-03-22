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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV

def optimize():
    print("--- THRESHOLD OPTIMIZATION SCAN ---")
    df = pd.read_csv("backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, inplace=True, errors='ignore')

    # Feature Engineering
    df['IncomePerAge'] = df['MonthlyIncome'] / (df['Age'] + 1)
    df['StabilityMetric'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    # Simple Robust Pipeline
    num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
    cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, [c for c in X.columns if c not in ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']]), ('cat', cat_pipeline, ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField'])])
    
    rf = RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample', random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=1000, scale_pos_weight=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb_model)], voting='soft')
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('classifier', ensemble)
    ])
    
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    best_acc = 0
    best_t = 0
    best_f1 = 0
    
    print("\nScanning Thresholds...")
    for t in np.arange(0.3, 0.85, 0.01):
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
            best_f1 = f1
            
    print(f"\nMax Accuracy: {best_acc:.4f} at Threshold {best_t:.2f}")
    print(f"Associated F1: {best_f1:.4f}")

if __name__ == "__main__":
    optimize()
