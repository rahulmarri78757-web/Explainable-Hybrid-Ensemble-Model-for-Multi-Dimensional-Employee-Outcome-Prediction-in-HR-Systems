import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")
DATASET_PATH = 'backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv'

def calculate_baseline_maes():
    print("Calculating Baseline MAEs for Performance Prediction...")
    df = pd.read_csv(DATASET_PATH)
    
    # Cleaning
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
    
    # Feature Engineering
    df['OverallSatisfaction'] = (
        df['JobSatisfaction'] * 0.4 +
        df['EnvironmentSatisfaction'] * 0.35 +
        df['RelationshipSatisfaction'] * 0.25
    )
    df['JobEngagement'] = df['JobInvolvement'] * df['PerformanceRating']
    
    X = df.drop('PerformanceRating', axis=1)
    y = df['PerformanceRating']
    
    cat_cols = X.select_dtypes(include=['object']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    results = {}
    
    # Linear Regression (Paper's choice)
    lr = LinearRegression()
    lr.fit(X_train_proc, y_train)
    results['Linear Regression'] = mean_absolute_error(y_test, lr.predict(X_test_proc))
    
    # Support Vector Regression (SVR)
    svr = SVR()
    svr.fit(X_train_proc, y_train)
    results['SVR'] = mean_absolute_error(y_test, svr.predict(X_test_proc))
    
    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_proc, y_train)
    results['Random Forest'] = mean_absolute_error(y_test, rf.predict(X_test_proc))
    
    print("\nRESULTS (Lower MAE is better):")
    for model, mae in results.items():
        print(f"  {model:<20}: {mae:.4f}")

if __name__ == "__main__":
    calculate_baseline_maes()
