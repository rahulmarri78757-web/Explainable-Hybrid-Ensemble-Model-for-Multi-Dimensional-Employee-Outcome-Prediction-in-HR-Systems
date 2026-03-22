"""
Optimize K-Means Clustering Metrics - Feature Engineering Experiments
This script tests different feature combinations to achieve better clustering metrics.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configuration
RANDOM_STATE = 42
N_CLUSTERS = 3
DATASET_PATH = 'backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv'

print("=" * 70)
print("TESTING FEATURE COMBINATIONS FOR OPTIMAL CLUSTERING")
print("=" * 70)

# Load dataset
df = pd.read_csv(DATASET_PATH)
print(f"\nDataset: {len(df)} employees")

# Set random seed for reproducibility
np.random.seed(RANDOM_STATE)

# Create base features
df['SentimentScore'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction']) / 8.0 + np.random.normal(0, 0.1, len(df))
df['SentimentScore'] = df['SentimentScore'].clip(0, 1)

df['WorkHours'] = (df['OverTime'].map({'Yes': 50, 'No': 40}) + 
                    df['DistanceFromHome'] * 0.5 + 
                    df['YearsAtCompany'] * 0.3)

# Test different feature combinations
feature_sets = [
    {
        'name': 'Original (4 features)',
        'features': ['WorkHours', 'SentimentScore', 'JobSatisfaction', 'WorkLifeBalance']
    },
    {
        'name': 'Extended Work Metrics (5 features)',
        'features': ['WorkHours', 'SentimentScore', 'JobSatisfaction', 'WorkLifeBalance', 'YearsAtCompany']
    },
    {
        'name': 'Satisfaction Focus (5 features)',
        'features': ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'JobInvolvement', 'RelationshipSatisfaction']
    },
    {
        'name': 'Balanced Profile (6 features)',
        'features': ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'MonthlyIncome', 'YearsAtCompany', 'JobInvolvement']
    },
    {
        'name': 'Comprehensive (7 features)',
        'features': ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'MonthlyIncome', 
                     'YearsAtCompany', 'JobInvolvement', 'RelationshipSatisfaction']
    }
]

results = []

print("\n" + "-" * 70)
print("TESTING FEATURE COMBINATIONS")
print("-" * 70)

for feature_set in feature_sets:
    name = feature_set['name']
    features = feature_set['features']
    
    X = df[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Compute metrics
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    
    results.append({
        'name': name,
        'features': features,
        'silhouette': sil,
        'davies_bouldin': db,
        'score': sil - db  # Combined score (higher silhouette, lower DB is better)
    })
    
    print(f"\n{name}:")
    print(f"  Silhouette Score:      {sil:.6f}")
    print(f"  Davies-Bouldin Index:  {db:.6f}")

# Find best combination
best = max(results, key=lambda x: x['score'])

print("\n" + "=" * 70)
print("BEST FEATURE COMBINATION")
print("=" * 70)
print(f"\nConfiguration: {best['name']}")
print(f"Features: {best['features']}")
print(f"\nSilhouette Score:      {best['silhouette']:.6f}")
print(f"Davies-Bouldin Index:  {best['davies_bouldin']:.6f}")
print("\n" + "=" * 70)
