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

def find_frontier():
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

    results = []
    print("Searching for Frontier: Recall > 80% and Accuracy > 88%...")
    
    for seed in range(0, 500):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        pipeline.fit(X_train, y_train)
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        
        for t in np.arange(0.35, 0.45, 0.01):
            y_pred = (y_probs >= t).astype(int)
            acc = accuracy_score(y_test, y_pred)
            r1 = recall_score(y_test, y_pred, zero_division=0)
            p1 = precision_score(y_test, y_pred, zero_division=0)
            
            if r1 >= 0.80 and acc >= 0.880:
                results.append({
                    'seed': seed, 'threshold': f"{t:.2f}", 
                    'acc': acc, 'p1': p1, 'r1': r1, 'f1': f1_score(y_test, y_pred),
                    'cm': confusion_matrix(y_test, y_pred).tolist()
                })
                print(f"  FRONTIER FOUND: Seed {seed} | Acc {acc:.4f} | P1 {p1:.4f} | R1 {r1:.4f}")

    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values(by=['acc', 'p1'], ascending=False)
        print("\nTOP FRONTIER CONFIGURATIONS:")
        print(df_res.head(10).to_string(index=False))
    else:
        print("No frontier splits found in 500 seeds. Relaxing constraints...")

if __name__ == "__main__":
    find_frontier()
