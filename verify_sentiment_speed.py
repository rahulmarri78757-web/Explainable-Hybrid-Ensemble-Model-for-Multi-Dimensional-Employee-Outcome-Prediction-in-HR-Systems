import time
import numpy as np
from backend.app.ml.prediction import ml_service

def verify_sentiment_speed():
    print("--- Sentiment Analysis Speed Test (VADER) ---")
    
    texts = [
        "The work environment is amazing and I love my team.",
        "Management is terrible and I want to leave immediately.",
        "It's okay, but the pay could be better.",
        "I am very happy with the growth opportunities here.",
        "Toxic culture and bad leadership."
    ] * 200 # 1000 texts
    
    print(f"Processing {len(texts)} texts...")
    
    latencies = []
    
    for text in texts:
        res = ml_service.analyze_sentiment(text)
        latencies.append(res['inference_time_ms'])
        
    avg = np.mean(latencies)
    total = np.sum(latencies)
    p95 = np.percentile(latencies, 95)
    
    print(f"\nAvg Latency: {avg:.4f} ms")
    print(f"95th Percentile: {p95:.4f} ms")
    print(f"Total Time for 1000 docs: {total:.2f} ms")
    print(f"Throughput: {1000 / (total/1000):.2f} docs/sec")

if __name__ == "__main__":
    verify_sentiment_speed()
