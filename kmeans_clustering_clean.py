"""
K-Means Clustering (K=3) for Burnout Risk Profiling - Clean Production Version

This script performs reproducible K-Means clustering with proper feature scaling
on the IBM HR Analytics dataset for burnout risk profiling.

Key Features:
- Proper StandardScaler normalization
- random_state=42 for reproducibility
- Silhouette Score and Davies-Bouldin Index metrics
- Optimized feature engineering for better cluster separation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
N_CLUSTERS = 3
DATASET_PATH = 'backend/app/ml/WA_Fn-UseC_-HR-Employee-Attrition.csv'

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("=" * 70)
    print("K-MEANS CLUSTERING FOR BURNOUT RISK PROFILING (K=3)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - K (clusters): {N_CLUSTERS}")
    print(f"  - Random state: {RANDOM_STATE} (ensures reproducibility)")
    print(f"  - Dataset: {DATASET_PATH}")
    
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: DATA LOADING")
    print("-" * 70)
    
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {len(df)} employees")
    
    # Set global random seed for reproducibility
    np.random.seed(RANDOM_STATE)
    
    # ========================================================================
    # FEATURE ENGINEERING - Optimized for Better Clustering
    # ========================================================================
    print("\n" + "-" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("-" * 70)
    
    # Create composite features that better capture burnout dimensions
    
    # Enhanced feature engineering for better cluster separation
    # Based on burnout research: satisfaction, work-life balance, tenure, and compensation
    
    # 1. Overall Satisfaction Score (weighted composite) - PRIMARY BURNOUT INDICATOR
    df['OverallSatisfaction'] = (
        df['JobSatisfaction'] * 0.4 +
        df['EnvironmentSatisfaction'] * 0.35 +
        df['RelationshipSatisfaction'] * 0.25
    )
    
    # 2. Work-Life Balance Score (critical burnout predictor)
    df['WLB_Score'] = df['WorkLifeBalance']
    
    # 3. Tenure Stress Index (years at company relative to total experience)
    df['TenureStress'] = (
        df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    )
    
    # 4. Income Satisfaction (normalized monthly income)
    df['IncomeSatisfaction'] = df['MonthlyIncome'] / 10000.0
    
    # 5. Job Involvement & Engagement
    df['JobEngagement'] = (
        df['JobInvolvement'] * 0.7 +
        df['JobLevel'] * 0.3
    )
    
    # Select optimized feature set (5 features for better separation)
    clustering_features = [
        'OverallSatisfaction',
        'WLB_Score',
        'TenureStress',
        'IncomeSatisfaction',
        'JobEngagement'
    ]
    
    X_cluster = df[clustering_features].copy()
    
    print(f"Engineered features: {len(clustering_features)}")
    for feat in clustering_features:
        print(f"  - {feat}")
    print(f"\nFeature matrix shape: {X_cluster.shape}")
    
    # ========================================================================
    # FEATURE SCALING
    # ========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: FEATURE SCALING")
    print("-" * 70)
    
    # Use StandardScaler for proper normalization (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    print("Applied StandardScaler normalization")
    print("  - Mean: 0.0")
    print("  - Std Dev: 1.0")
    print(f"Scaled matrix shape: {X_scaled.shape}")
    
    # ========================================================================
    # K-MEANS CLUSTERING
    # ========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: K-MEANS CLUSTERING")
    print("-" * 70)
    
    # Initialize KMeans with optimal parameters for better convergence
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=300,          # More initializations for better stability
        max_iter=1000,       # Sufficient iterations for convergence
        algorithm='lloyd'    # Classic K-Means algorithm
    )
    
    # Fit and predict cluster labels
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    print(f"K-Means clustering completed")
    print(f"  - Converged in {kmeans.n_iter_} iterations")
    print(f"  - Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    
    # Show cluster distribution
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("\nCluster Distribution:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(cluster_labels)) * 100
        print(f"  Cluster {cluster_id}: {count:4d} employees ({percentage:5.1f}%)")
    
    # ========================================================================
    # EVALUATION METRICS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: CLUSTERING EVALUATION METRICS")
    print("=" * 70)
    
    # Compute Silhouette Score
    silhouette = silhouette_score(X_scaled, cluster_labels)
    
    # Compute Davies-Bouldin Index
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
    
    # ========================================================================
    # FINAL RESULTS
    # ========================================================================
    print("\n" + "-" * 70)
    print("FINAL NUMERIC SCORES")
    print("-" * 70)
    print(f"\nSilhouette Score:      {silhouette:.6f}")
    print(f"Davies-Bouldin Index:  {davies_bouldin:.6f}")
    
    # ========================================================================
    # METRIC INTERPRETATIONS
    # ========================================================================
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
    
    # ========================================================================
    # QUALITY ASSESSMENT
    # ========================================================================
    print("\n" + "=" * 70)
    print("CLUSTERING QUALITY ASSESSMENT")
    print("=" * 70)
    
    # Silhouette interpretation
    if silhouette >= 0.51:
        sil_quality = "MODERATE-TO-STRONG"
        sil_detail = "Clusters show reasonable to good separation"
    elif silhouette >= 0.26:
        sil_quality = "WEAK-TO-MODERATE"
        sil_detail = "Clusters have some structure but overlap exists"
    else:
        sil_quality = "INSUFFICIENT"
        sil_detail = "No clear cluster structure detected"
    
    # Davies-Bouldin interpretation
    if davies_bouldin < 1.0:
        db_quality = "GOOD"
        db_detail = "Within-cluster cohesion exceeds between-cluster separation"
    elif davies_bouldin < 2.0:
        db_quality = "MODERATE"
        db_detail = "Acceptable cluster cohesion with some overlap"
    else:
        db_quality = "POOR"
        db_detail = "Significant cluster overlap detected"
    
    print(f"\nSilhouette Assessment: {sil_quality}")
    print(f"  → {sil_detail}")
    
    print(f"\nDavies-Bouldin Assessment: {db_quality}")
    print(f"  → {db_detail}")
    
    # Overall verdict
    print("\n" + "-" * 70)
    if silhouette >= 0.40 and davies_bouldin < 1.2:
        print("Overall: ACCEPTABLE FOR RESEARCH")
        print("The clustering reveals empirically distinct burnout risk profiles")
        print("suitable for exploratory analysis and profiling.")
    elif silhouette >= 0.26 or davies_bouldin < 1.5:
        print("Overall: MARGINAL BUT USABLE")
        print("Clustering shows weak structure. Consider feature refinement.")
    else:
        print("Overall: NOT RECOMMENDED")
        print("Clustering quality is insufficient. Consider alternative approaches.")
    
    print("\n" + "=" * 70)
    print("CLUSTERING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'cluster_labels': cluster_labels,
        'n_clusters': N_CLUSTERS
    }

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    results = main()
