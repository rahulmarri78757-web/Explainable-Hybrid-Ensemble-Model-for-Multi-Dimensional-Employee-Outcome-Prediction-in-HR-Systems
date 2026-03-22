import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")
DATASET_PATH = 'backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv'

def calculate_safe_mae():
    df = pd.read_csv(DATASET_PATH)
    
    # Cleaning
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    
    # NO LEAKAGE FEATURE ENGINEERING
    # Only use factors that are known BEFORE the performance review
    df['OverallSatisfaction'] = (
        df['JobSatisfaction'] * 0.4 +
        df['EnvironmentSatisfaction'] * 0.35 +
        df['RelationshipSatisfaction'] * 0.25
    )
    # Removed: JobEngagement = Involvement * PerformanceRating (Target Leakage!)
    
    X = df.drop('PerformanceRating', axis=1)
    y = df['PerformanceRating']
    
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    # Test across multiple seeds to be extra safe
    maes = []
    for seed in range(0, 10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_proc, y_train)
        maes.append(mean_absolute_error(y_test, model.predict(X_test_proc)))
    
    print(f"Safe MAE (Average across 10 seeds): {np.mean(maes):.4f}")

if __name__ == "__main__":
    calculate_safe_mae()
