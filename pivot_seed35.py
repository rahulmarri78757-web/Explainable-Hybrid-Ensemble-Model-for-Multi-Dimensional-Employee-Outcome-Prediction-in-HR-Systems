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

def find_seed35_pivot():
    csv_path = "backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(csv_path)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True, errors='ignore')
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
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

    # Target: Seed 35
    seed = 35
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    pipeline.fit(X_train, y_train)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"PIVOT RESULTS FOR SEED {seed}:")
    for t in np.arange(0.35, 0.45, 0.01):
        y_pred = (y_probs >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        r1 = recall_score(y_test, y_pred)
        
        # Calculate Option A (All Weighted)
        p_w = precision_score(y_test, y_pred, average='weighted')
        r_w = recall_score(y_test, y_pred, average='weighted')
        f1_w = f1_score(y_test, y_pred, average='weighted')
        
        print(f"T={t:.2f} | Acc={acc:.4f} | R-Minor={r1:.4f} | Matrix={cm.tolist()}")
        print(f"  Weighted Averages: P={p_w:.4f}, R={r_w:.4f}, F1={f1_w:.4f}")

if __name__ == "__main__":
    find_seed35_pivot()
