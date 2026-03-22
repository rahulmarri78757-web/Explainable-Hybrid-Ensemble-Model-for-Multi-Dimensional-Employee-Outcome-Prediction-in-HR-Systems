import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the IBM HR Dataset
csv_path = 'backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = pd.read_csv(csv_path)

# Create synthetic sentiment scores based on job satisfaction and environment satisfaction
np.random.seed(42)
df['SentimentScore'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction']) / 8.0 + np.random.normal(0, 0.1, len(df))
df['SentimentScore'] = df['SentimentScore'].clip(0, 1)

# Select features for burnout clustering
# WorkHours approximated by: overtime + distance from home + years at company
df['WorkHours'] = (df['OverTime'].map({'Yes': 50, 'No': 40}) + 
                   df['DistanceFromHome'] * 0.5 + 
                   df['YearsAtCompany'] * 0.3)

# Clustering features
clustering_features = ['WorkHours', 'SentimentScore', 'JobSatisfaction', 'WorkLifeBalance']
X_cluster = df[clustering_features].copy()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("=" * 60)
print("CLUSTERING VALIDATION METRICS (Raw Computation)")
print("=" * 60)
print(f"Dataset size: {len(df)} employees")
print(f"Features used: {clustering_features}")
print("-" * 60)

results = []

# Test k = 2, 3, 4
for k in [2, 3, 4]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Compute metrics
    sil_score = silhouette_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)
    
    results.append({
        'k': k,
        'silhouette': sil_score,
        'davies_bouldin': db_index
    })
    
    print(f"\nk = {k}:")
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Davies-Bouldin Index: {db_index:.4f}")

# Find best k (highest silhouette, lowest DB)
best_by_silhouette = max(results, key=lambda x: x['silhouette'])
best_by_db = min(results, key=lambda x: x['davies_bouldin'])

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)

if best_by_silhouette['k'] == best_by_db['k']:
    best_k = best_by_silhouette['k']
    print(f"Best k: {best_k} (agreed by both metrics)")
else:
    print(f"Best k by Silhouette: {best_by_silhouette['k']}")
    print(f"Best k by Davies-Bouldin: {best_by_db['k']}")
    # Use silhouette as primary metric if metrics disagree
    best_k = best_by_silhouette['k']
    print(f"Recommended k: {best_k} (prioritizing highest Silhouette)")

best_result = [r for r in results if r['k'] == best_k][0]
print(f"\nFinal Metrics (k={best_k}):")
print(f"  Silhouette Score: {best_result['silhouette']:.4f}")
print(f"  Davies-Bouldin Index: {best_result['davies_bouldin']:.4f}")
print("\nInterpretation: The clusters represent distinct behavioral risk profiles")
print("for burnout detection based on work patterns and sentiment indicators.")
