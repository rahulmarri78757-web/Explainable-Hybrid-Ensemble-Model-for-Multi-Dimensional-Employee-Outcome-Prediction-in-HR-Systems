
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configuration
DATASET_PATH = 'backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv'
RANDOM_STATE = 42

def report_baselines():
    print("="*60)
    print("BASELINE MODEL REPORT (Honest Evaluation)")
    print("="*60)
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    # 2. Preprocessing
    # Drop useless cols
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    
    # Target
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    
    # Transformers
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    # 3. Define Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced'),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_STATE, scale_pos_weight=5) # scale_pos roughly matches imbalance
    }
    
    # Store trained models for ensemble
    trained_estimators = []
    
    print(f"{'MODEL':<25} | {'ACCURACY':<10} | {'F1-SCORE':<10} | {'RECALL':<10}")
    print("-" * 65)
    
    # 4. Evaluate Individual Models
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        
        print(f"{name:<25} | {acc*100:.2f}%     | {f1*100:.2f}%     | {rec*100:.2f}%")
        
        # Save for voting (must use the fitted estimator steps)
        # Note: VotingClassifier needs (name, estimator). We need to verify if we can just pass the pipelines.
        # Ideally, we put the voting classifier INSIDE a pipeline, or VotingClassifier with pipelines as estimators.
        trained_estimators.append((name, pipeline))

    # 5. Hybrid Ensemble
    print("-" * 65)
    
    # Voting Ensemble using the *Pipelines* as estimators
    ensemble = VotingClassifier(
        estimators=trained_estimators,
        voting='soft'
    )
    
    # Note: VotingClassifier with Pipelines usually requires 'fit' to be called on the ensemble itself
    # unless we use fitted estimators with appropriate wrapper (which is complex).
    # Easier: Just re-fit the ensemble of pipelines.
    
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    
    ens_acc = accuracy_score(y_test, y_pred_ens)
    ens_f1 = f1_score(y_test, y_pred_ens)
    ens_rec = recall_score(y_test, y_pred_ens)
    
    print(f"{'Hybrid Ensemble':<25} | {ens_acc*100:.2f}%     | {ens_f1*100:.2f}%     | {ens_rec*100:.2f}%")
    print("="*60)
    
    # 6. Consistency Check
    # "Consistent" usually means: Is the Ensemble better than the best single model?
    # And are the baselines within a reasonable range (e.g. > 80%)?
    
    best_single = max([accuracy_score(y_test, m[1].predict(X_test)) for m in trained_estimators])
    lift = ens_acc - best_single
    
    print("\nCONSISTENCY ANALYSIS:")
    print(f"Best Single Model Acc: {best_single*100:.2f}%")
    print(f"Ensemble Lift:         {lift*100:+.2f}% points")
    
    if lift > 0:
        print("RESULT: Consistent. Ensemble improves upon single models.")
    else:
        print("RESULT: Inconsistent/Neutral. Ensemble did not outperform best single model.")
        
    if ens_acc < 0.90:
        print(f"NOTE: Current Hybrid Accuracy ({ens_acc*100:.2f}%) is significantly below target (94.78%).")
        print("      Calibration/Optimization is required to reach target.")

if __name__ == "__main__":
    report_baselines()
