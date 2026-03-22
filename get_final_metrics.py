import sys
import os
import time
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

try:
    from app.ml.prediction import ml_service
    
    print("-" * 30)
    print("REVIEWER OPTIMIZATION METRICS")
    print("-" * 30)
    
    # 1. Stratified Accuracy
    metrics = ml_service._evaluate_model_performance("attrition")
    print(f"1. Stratified test accuracy: {metrics['accuracy']:.2f}%")
    
    # 2 & 3. CV Results
    overall = ml_service.get_model_metrics()
    cv = overall['cv_info']
    print(f"2. CV mean accuracy: {cv['mean']:.2f}%")
    print(f"3. CV std deviation: {cv['std']:.2f}%")
    
    # 4. Burnout Consistency
    burnout = ml_service._evaluate_model_performance("burnout")
    print(f"4. Burnout consistency value: {burnout['behavioral_consistency']:.2f}%")
    
    # 5. Bias Flagged % (Simulation)
    bias_flags = []
    for _ in range(500):
        # Mixed population (some biased, some not)
        if np.random.random() > 0.22:
            ratings = np.random.normal(3.2, 0.6, 15).clip(1, 5).tolist()
        else:
            ratings = np.random.normal(4.5, 0.2, 15).clip(1, 5).tolist()
        res = ml_service.detect_managerial_bias(ratings)
        bias_flags.append(1 if res['bias_detected'] else 0)
    
    print(f"5. Bias flagged %: {np.mean(bias_flags) * 100:.1f}%")
    
    # 6. Inference Time
    times = []
    for _ in range(100):
        start = time.perf_counter()
        ml_service.predict_attrition([10, 0, 3, 24, 5])
        times.append(time.perf_counter() - start)
    
    print(f"6. Average inference time: {np.mean(times) * 1000:.2f}ms")
    print("-" * 30)

except Exception as e:
    print(f"Error collecting metrics: {e}")
    import traceback
    traceback.print_exc()
