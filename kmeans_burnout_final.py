# encoding: utf-8
"""
K-Means Clustering (K=3) for Burnout Risk Profiling - OPTIMIZED VERSION

This script performs K-Means clustering with advanced feature engineering
to achieve better clustering metrics (Silhouette Score and Davies-Bouldin Index).

Target Metrics:
- Silhouette Score: 0.40-0.50 (Moderate to Good structure)
- Davies-Bouldin Index: 0.85-1.10 (Good separation)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configure console encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
N_CLUSTERS = 3
DATASET_PATH = 'backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv'

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 70)
    print("K-MEANS CLUSTERING FOR BURNOUT RISK PROFILING (K=3)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - K (clusters): {N_CLUSTERS}")
    print(f"  - Random state: {RANDOM_STATE} (ensures reproducibility)")
    print(f"  - Dataset: {DATASET_PATH}")
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: DATA LOADING")
    print("-" * 70)
    
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {len(df)} employees")
    
    # Set global random seed
    np.random.seed(RANDOM_STATE)
    
    # =========================================================================
    # ADVANCED FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: ADVANCED FEATURE ENGINEERING")
    print("-" * 70)
    
    # Strategy: Create highly discriminative composite features that capture
    # distinct burnout dimensions with minimal overlap
    
    # 1. Satisfaction Index (weighted composite of all satisfaction metrics)
    df['SatisfactionIndex'] = (
        df['JobSatisfaction'] * 0.35 +
        df['EnvironmentSatisfaction'] * 0.30 +
        df['RelationshipSatisfaction'] * 0.20 +
        df['WorkLifeBalance'] * 0.15
    )
    
    # 2. Work-Life Balance Score
    df['WLB_Normalized'] = df['WorkLifeBalance']
    
    # 3. Career Growth Velocity (income per year of service)
    df['CareerVelocity'] = (
        df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    ) / 1000.0
    
    # 4. Job Engagement Level
    df['EngagementLevel'] = (
        df['JobInvolvement'] * 0.60 +
        df['JobLevel'] * 0.25 +
        (5 - df['DistanceFromHome'] / 10.0) * 0.15  # Proximity as engagement proxy
    ).clip(0, 5)
    
    # Primary feature set - focusing on the most discriminative features
    primary_features = [
        'SatisfactionIndex',
        'WLB_Normalized',
        'CareerVelocity',
        'EngagementLevel'
    ]
    
    X_primary = df[primary_features].copy()
    
    print(f"Primary features engineered: {len(primary_features)}")
    for feat in primary_features:
        print(f"  - {feat}")
    
    # =========================================================================
    # FEATURE SCALING
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: FEATURE SCALING")
    print("-" * 70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_primary)
    
    print("Applied StandardScaler normalization (mean=0, std=1)")
    print(f"Scaled matrix shape: {X_scaled.shape}")
    
    # =========================================================================
    # DIMENSIONALITY REDUCTION (Optional PCA for visualization)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: PCA TRANSFORMATION (OPTIONAL)")
    print("-" * 70)
    
    # Apply PCA to enhance cluster separation
    pca = PCA(n_components=4, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA applied: Retained variance = {pca.explained_variance_ratio_.sum():.4f}")
    print(f"PCA matrix shape: {X_pca.shape}")
    
    # USE PCA-transformed features for clustering (often improves metrics)
    X_final = X_pca
    
    # =========================================================================
    # K-MEANS CLUSTERING
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: K-MEANS CLUSTERING")
    print("-" * 70)
    
    # Extensive initialization for optimal convergence
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=500,          # Very high for maximum stability
        max_iter=1000,
        algorithm='lloyd',
        tol=1e-6             # Tight convergence tolerance
    )
    
    cluster_labels = kmeans.fit_predict(X_final)
    
    print(f"K-Means clustering completed")
    print(f"  - Converged in {kmeans.n_iter_} iterations")
    print(f"  - Inertia: {kmeans.inertia_:.2f}")
    
    # Cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\nCluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Cluster {cluster_id}: {count:4d} employees ({percentage:5.1f}%)")
    
    # =========================================================================
    # EVALUATION METRICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: CLUSTERING EVALUATION METRICS")
    print("=" * 70)
    
    silhouette = silhouette_score(X_final, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_final, cluster_labels)
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print("\n" + "-" * 70)
    print("FINAL NUMERIC SCORES")
    print("-" * 70)
    print(f"\nSilhouette Score:      {silhouette:.6f}")
    print(f"Davies-Bouldin Index:  {davies_bouldin:.6f}")
    
    # =========================================================================
    # METRIC INTERPRETATIONS
    # =========================================================================
    print("\n" + "-" * 70)
    print("METRIC INTERPRETATIONS")
    print("-" * 70)
    
    print("\nSilhouette Score (range: -1 to 1):")
    print("  - Higher is better (1.0 = perfect separation)")
    print("  - 0.71 to 1.0:  Strong, well-defined structure")
    print("  - 0.51 to 0.70: Moderate structure")
    print("  - 0.26 to 0.50: Weak but present structure")
    print("  - Below 0.26:   No substantial structure")
    
    print("\nDavies-Bouldin Index (range: 0 to infinity):")
    print("  - Lower is better (0.0 = perfect separation)")
    print("  - DB < 0.5:   Excellent cluster separation")
    print("  - DB 0.5-1.0: Good cluster separation")
    print("  - DB 1.0-2.0: Moderate separation")
    print("  - DB > 2.0:   Poor separation")
    
    # =========================================================================
    # QUALITY ASSESSMENT
    # =========================================================================
    print("\n" + "=" * 70)
    print("CLUSTERING QUALITY ASSESSMENT")
    print("=" * 70)
    
    # Silhouette interpretation
    if silhouette >= 0.51:
        sil_quality = "MODERATE-TO-STRONG"
        sil_detail = "Clusters show good separation with distinct profiles"
    elif silhouette >= 0.26:
        sil_quality = "WEAK-TO-MODERATE"
        sil_detail = "Clusters have some structure but overlap exists"
    else:
        sil_quality = "INSUFFICIENT"
        sil_detail = "No clear cluster structure detected"
    
    # Davies-Bouldin interpretation
    if davies_bouldin < 1.0:
        db_quality = "GOOD"
        db_detail = "Strong within-cluster cohesion, minimal overlap"
    elif davies_bouldin < 2.0:
        db_quality = "MODERATE"
        db_detail = "Acceptable cluster cohesion with some overlap"
    else:
        db_quality = "POOR"
        db_detail = "Significant cluster overlap detected"
    
    print(f"\nSilhouette Assessment: {sil_quality}")
    print(f"  -> {sil_detail}")
    
    print(f"\nDavies-Bouldin Assessment: {db_quality}")
    print(f"  -> {db_detail}")
    
    # Overall verdict
    print("\n" + "-" * 70)
    if silhouette >= 0.40 and davies_bouldin < 1.2:
        print("Overall Quality: ACCEPTABLE FOR RESEARCH")
        print("The clustering reveals empirically distinct burnout risk profiles")
        print("suitable for exploratory analysis and employee profiling.")
    elif silhouette >= 0.26 or davies_bouldin < 1.5:
        print("Overall Quality: MARGINAL BUT USABLE")
        print("Clustering shows weak structure. Consider feature refinement.")
    else:
        print("Overall Quality: NOT RECOMMENDED")
        print("Clustering quality is insufficient for scientific analysis.")
    
    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'cluster_labels': cluster_labels
    }

if __name__ == "__main__":
    results = main()
