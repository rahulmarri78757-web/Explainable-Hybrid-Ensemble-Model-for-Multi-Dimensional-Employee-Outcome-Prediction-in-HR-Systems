
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from app.ml.prediction import ml_service
    
    print("Initialize ML Service...")
    
    # Force re-init if needed? The import usually instantiates it.
    # But let's check what we got.
    
    metrics = ml_service.get_model_metrics()
    
    print("\n--- FINAL METRICS FROM PRODUCTION CODE ---")
    print("Attrition Prediction:")
    for k, v in metrics['Attrition Prediction'].items():
        print(f"  {k}: {v:.2f}")

    print("\nSentiment Analysis:")
    if 'Sentiment Analysis' in metrics:
        for k, v in metrics['Sentiment Analysis'].items():
            print(f"  {k}: {v:.2f}")
    else:
        print("  [MISSING]")
        
    print("\nClustering:")
    for k, v in metrics['clustering'].items():
        print(f"  {k}: {v:.4f}")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
