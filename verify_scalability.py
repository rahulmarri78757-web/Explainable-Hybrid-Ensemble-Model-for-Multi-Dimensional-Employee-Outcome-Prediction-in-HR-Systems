import time
import concurrent.futures
import numpy as np
import pandas as pd
from backend.app.ml.prediction import ml_service

def simulate_concurrency(num_users=100):
    print(f"--- Scalability Simulation: {num_users} Concurrent Users ---")
    print("Hardware: CPU-based local machine (simulated production node)")
    
    # Sample features for prediction
    sample_features = [12, 0.5, 3, 24, 15] # OverTime, JobSat, Perf, YearsAtCo, SalaryHike
    
    latencies = []
    errors = 0
    
    def single_request():
        try:
            start = time.time()
            ml_service.predict_attrition(sample_features)
            return (time.time() - start) * 1000 # to ms
        except Exception:
            return None
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [executor.submit(single_request) for _ in range(num_users)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is None:
                errors += 1
            else:
                latencies.append(res)
    total_time = time.time() - start_time

    avg_latency = np.mean(latencies) if latencies else 0
    max_latency = np.max(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0
    error_rate = (errors / num_users) * 100
    
    print("\n[Results]")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"Max Latency: {max_latency:.2f} ms")
    print(f"95th Percentile: {p95_latency:.2f} ms")
    print(f"Error Rate: {error_rate:.1f}%")
    print(f"Total Test Time: {total_time:.2f}s")
    
    return {
        "avg": avg_latency,
        "max": max_latency,
        "error_rate": error_rate
    }

if __name__ == "__main__":
    simulate_concurrency()
