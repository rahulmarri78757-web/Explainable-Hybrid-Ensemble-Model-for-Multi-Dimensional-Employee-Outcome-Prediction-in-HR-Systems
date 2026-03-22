import time
from backend.app.ml.prediction import ml_service

def verify_single_latency():
    print("--- Single Prediction Latency ---")
    sample_features = [12, 0.5, 3, 24, 15]
    
    # Warmup
    ml_service.predict_attrition(sample_features)
    
    latencies = []
    for _ in range(50):
        start = time.time()
        ml_service.predict_attrition(sample_features)
        latencies.append((time.time() - start) * 1000)
        
    print(f"Avg Latency (No Load): {sum(latencies)/len(latencies):.2f} ms")
    print(f"Min Latency: {min(latencies):.2f} ms")
    print(f"Max Latency: {max(latencies):.2f} ms")

if __name__ == "__main__":
    verify_single_latency()
