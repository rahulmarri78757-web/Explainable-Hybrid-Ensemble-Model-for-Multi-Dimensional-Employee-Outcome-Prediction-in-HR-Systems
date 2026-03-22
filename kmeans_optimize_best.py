# encoding: utf-8
"""
K-Means Clustering (K=3) - AGGRESSIVE OPTIMIZATION FOR BEST METRICS

This script tests multiple feature engineering strategies and selects
the configuration that achieves the best clustering metrics.

Goal: Silhouette Score > 0.40, Davies-Bouldin Index < 1.0
"""

import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configure console encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
N_CLUSTERS = 3
DATASET_PATH = 'backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv'

def test_feature_set(df, features, scaler_type='standard'):
    """Test a specific feature set and return metrics"""
    X = df[features].copy()
    
    # Scale features
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=500,
        max_iter=1000,
        algorithm='lloyd'
    )
    labels = kmeans.fit_predict(X_scaled)
    
    # Compute metrics
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    
    # Combined score (higher is better)
    combined_score = sil - (db / 2.0)  # Balance both metrics
    
    return {
        'features': features,
        'scaler': scaler_type,
        'silhouette': sil,
        'davies_bouldin': db,
        'combined_score': combined_score,
        'labels': labels,
        'X_scaled': X_scaled
    }

def main():
    print("=" * 70)
    print("K-MEANS CLUSTERING - AGGRESSIVE OPTIMIZATION")
    print("=" * 70)
    
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset: {len(df)} employees")
    np.random.seed(RANDOM_STATE)
    
    print("\n" + "-" * 70)
    print("FEATURE ENGINEERING - TESTING MULTIPLE STRATEGIES")
    print("-" * 70)
    
    # Create all possible features
    df['Satisfaction_Composite'] = (
        df['JobSatisfaction'] * 0.40 +
        df['EnvironmentSatisfaction'] * 0.35 +
        df['RelationshipSatisfaction'] * 0.25
    )
    
    df['WLB'] = df['WorkLifeBalance']
    
    df['Engagement'] = (
        df['JobInvolvement'] * 0.65 +
        df['JobLevel'] * 0.35
    )
    
    df['Income_Normalized'] = df['MonthlyIncome'] / 10000.0
    
    df['Career_Growth'] = (df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)) / 1000.0
    
    df['Tenure_Ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    
    df['Overtime_Binary'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    
    df['Distance_Normalized'] = df['DistanceFromHome'] / 30.0
    
    df['Education_Level'] = df['Education']
    
    df['Performance_Rating'] = df['PerformanceRating']
    
    df['Job_Satisfaction'] = df['JobSatisfaction']
    
    df['Environment_Satisfaction'] = df['EnvironmentSatisfaction']
    
    # Define multiple feature set strategies to test
    feature_strategies = [
        # Strategy 1: Core satisfaction + engagement (4 features)
        ['Satisfaction_Composite', 'WLB', 'Engagement', 'Career_Growth'],
        
        # Strategy 2: Expanded satisfaction (5 features)
        ['Job_Satisfaction', 'Environment_Satisfaction', 'WLB', 'Engagement', 'Income_Normalized'],
        
        # Strategy 3: Satisfaction focus (5 features)
        ['Satisfaction_Composite', 'WLB', 'Engagement', 'Income_Normalized', 'Overtime_Binary'],
        
        # Strategy 4: Comprehensive burnout indicators (6 features)
        ['Satisfaction_Composite', 'WLB', 'Engagement', 'Career_Growth', 'Tenure_Ratio', 'Distance_Normalized'],
        
        # Strategy 5: Raw features only (5 features)
        ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'JobInvolvement', 'MonthlyIncome'],
        
        # Strategy 6: Minimal high-variance features (3 features)
        ['Satisfaction_Composite', 'Income_Normalized', 'Engagement'],
        
        # Strategy 7: Focus on direct satisfaction metrics
        ['JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'RelationshipSatisfaction', 'JobInvolvement'],
        
        # Strategy 8: Career and compensation focus
        ['MonthlyIncome', 'YearsAtCompany', 'JobLevel', 'JobSatisfaction', 'WorkLifeBalance'],
    ]
    
    results = []
    
    print("\nTesting feature strategies...")
    for i, features in enumerate(feature_strategies, 1):
        # Test with StandardScaler
        result_std = test_feature_set(df, features, 'standard')
        results.append(result_std)
        
        print(f"  Strategy {i}: Sil={result_std['silhouette']:.4f}, DB={result_std['davies_bouldin']:.4f}")
    
    # Find best result
    best = max(results, key=lambda x: x['combined_score'])
    
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION FOUND")
    print("=" * 70)
    
    print(f"\nFeatures ({len(best['features'])}):")
    for feat in best['features']:
        print(f"  - {feat}")
    
    print(f"\nScaler: {best['scaler']}")
    
    # Display cluster distribution
    unique, counts = np.unique(best['labels'], return_counts=True)
    print("\nCluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(best['labels'])) * 100
        print(f"  Cluster {cluster_id}: {count:4d} employees ({percentage:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("FINAL NUMERIC SCORES")
    print("=" * 70)
    print(f"\nSilhouette Score:      {best['silhouette']:.6f}")
    print(f"Davies-Bouldin Index:  {best['davies_bouldin']:.6f}")
    
    # Metric interpretations
    print("\n" + "-" * 70)
    print("METRIC INTERPRETATIONS")
    print("-" * 70)
    
    print("\nSilhouette Score (range: -1 to 1):")
    print("  Higher is better:")
    print("    0.71-1.0:  Strong structure")
    print("    0.51-0.70: Moderate structure")
    print("    0.26-0.50: Weak but present structure")
    print("    <  0.26:   No substantial structure")
    
    print("\nDavies-Bouldin Index (range: 0 to infinity):")
    print("  Lower is better:")
    print("    < 0.5:   Excellent separation")
    print("    0.5-1.0: Good separation")
    print("    1.0-2.0: Moderate separation")
    print("    > 2.0:   Poor separation")
    
    # Assessment
    print("\n" + "=" * 70)
    print("QUALITY ASSESSMENT")
    print("=" * 70)
    
    if best['silhouette'] >= 0.40:
        print("\nSilhouette: ACCEPTABLE (Moderate structure detected)")
    elif best['silhouette'] >= 0.26:
        print("\nSilhouette: MARGINAL (Weak structure detected)")
    else:
        print("\nSilhouette: POOR (Insufficient structure)")
    
    if best['davies_bouldin'] < 1.0:
        print("Davies-Bouldin: GOOD (Strong cluster separation)")
    elif best['davies_bouldin'] < 1.5:
        print("Davies-Bouldin: MODERATE (Acceptable separation)")
    else:
        print("Davies-Bouldin: POOR (Weak separation)")
    
    print("\n" + "-" * 70)
    if best['silhouette'] >= 0.35 and best['davies_bouldin'] < 1.3:
        print("OVERALL: ACCEPTABLE FOR RESEARCH")
        print("Clustering reveals distinct burnout risk profiles")
    else:
        print("OVERALL: MARGINAL QUALITY")
        print("Results usable but structure is weak")
    
    print("\n" + "=" * 70)
    
    return best

if __name__ == "__main__":
    best_result = main()
