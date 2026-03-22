import os
import time
import concurrent.futures
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# --- 1. IBM DATASET EVALUATION ---
def evaluate_ibm():
    csv_path = "backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(csv_path)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    
    total_samples = len(df)
    class_counts = df['Attrition'].value_counts().to_dict()
    
    cat_features = ['JobRole', 'Department', 'Gender', 'BusinessTravel', 'OverTime', 'MaritalStatus', 'EducationField']
    num_features = [c for c in df.columns if c not in cat_features + ['Attrition']]
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # Baseline Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42),
        'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, scale_pos_weight=4, eval_metric='logloss', random_state=42, use_label_encoder=False)
    }
    
    baseline_results = []
    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_probs = pipe.predict_proba(X_test)[:, 1]
        baseline_results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_probs)
        })

    # Hybrid Ensemble
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in models.items()],
        voting='soft'
    )
    
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', ensemble)])
    full_pipeline.fit(X_train, y_train)
    
    y_pred = full_pipeline.predict(X_test)
    y_probs = full_pipeline.predict_proba(X_test)[:, 1]
    
    # 5-Fold CV
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=cv_strategy, scoring='f1')
    
    # Clustering
    cluster_data = preprocessor.fit_transform(X_train)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(cluster_data)
    sil = silhouette_score(cluster_data, kmeans.labels_)
    dbi = davies_bouldin_score(cluster_data, kmeans.labels_)
    
    # Scalability
    def single_req():
        start = time.time()
        full_pipeline.predict(X_test.iloc[[0]])
        return (time.time() - start) * 1000
    
    latencies = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(single_req) for _ in range(100)]
        for f in concurrent.futures.as_completed(futures):
            latencies.append(f.result())
            
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'info': {'total': total_samples, 'counts': class_counts},
        'metrics': {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_probs)
        },
        'cm': {'TN': cm[0,0], 'FP': cm[0,1], 'FN': cm[1,0], 'TP': cm[1,1]},
        'baselines': baseline_results,
        'cv': {'mean': np.mean(cv_scores), 'std': np.std(cv_scores)},
        'clustering': {'sil': sil, 'dbi': dbi},
        'scale': {'avg': np.mean(latencies), 'max': np.max(latencies)}
    }

# --- 2. EXTERNAL DATASET EVALUATION ---
def evaluate_external():
    csv_path = "backend/app/ml/job_change_data.csv"
    df = pd.read_csv(csv_path).dropna(subset=['target'])
    
    total_samples = len(df)
    class_counts = df['target'].value_counts().to_dict()
    
    cat_features = ['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job']
    num_features = ['city_development_index', 'training_hours']
    
    X = df.drop(['target', 'enrollee_id'], axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    from sklearn.impute import SimpleImputer
    num_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, num_features), ('cat', cat_pipeline, cat_features)])
    
    # Identical Architecture
    models = {
        'rf': RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42),
        'lr': LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42),
        'svc': SVC(probability=True, class_weight='balanced', random_state=42),
        'xgb': xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, scale_pos_weight=4, eval_metric='logloss', random_state=42, use_label_encoder=False)
    }
    
    ensemble = VotingClassifier(estimators=[(k, v) for k, v in models.items()], voting='soft')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', ensemble)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_strategy, scoring='f1')
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'info': {'total': total_samples, 'counts': class_counts},
        'metrics': {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_probs)
        },
        'cm': {'TN': cm[0,0], 'FP': cm[0,1], 'FN': cm[1,0], 'TP': cm[1,1]},
        'cv': {'mean': np.mean(cv_scores), 'std': np.std(cv_scores)}
    }

if __name__ == "__main__":
    ibm = evaluate_ibm()
    ext = evaluate_external()
    
    print("\n--- SECTION A: IBM HR DATASET ---")
    print(f"Total Samples: {ibm['info']['total']}")
    print(f"Class Distribution: {ibm['info']['counts']}")
    print(f"Metrics (Ensemble): {ibm['metrics']}")
    print(f"Confusion Matrix: {ibm['cm']}")
    print(f"CV F1: {ibm['cv']['mean']:.4f} ± {ibm['cv']['std']:.4f}")
    print(f"Clustering: {ibm['clustering']}")
    print(f"Scalability: {ibm['scale']}")
    print("\nBaseline Table:")
    print(pd.DataFrame(ibm['baselines']).to_string(index=False))
    
    print("\n--- SECTION B: EXTERNAL DATASET ---")
    print(f"Total Samples: {ext['info']['total']}")
    print(f"Class Distribution: {ext['info']['counts']}")
    print(f"Metrics (Ensemble): {ext['metrics']}")
    print(f"Confusion Matrix: {ext['cm']}")
    print(f"CV F1: {ext['cv']['mean']:.4f} ± {ext['cv']['std']:.4f}")
