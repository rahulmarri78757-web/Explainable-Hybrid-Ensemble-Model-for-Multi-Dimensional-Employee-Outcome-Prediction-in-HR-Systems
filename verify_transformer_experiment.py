import time
import pandas as pd
from transformers import pipeline
from backend.app.ml.prediction import ml_service

def verify_transformer_comparison():
    print("--- Transformer vs VADER Micro-Benchmark ---")
    
    # 1. Setup Data (Synthetic Ground Truth)
    data = [
        ("I love the culture and my team is amazing.", "POSITIVE"),
        ("Great opportunities for growth and learning.", "POSITIVE"),
        ("Management is supportive and listens to feedback.", "POSITIVE"),
        ("The benefits are excellent and pay is good.", "POSITIVE"),
        ("I feel valued and appreciated here.", "POSITIVE"),
        ("Toxic environment and terrible leadership.", "NEGATIVE"),
        ("I am underpaid and overworked.", "NEGATIVE"),
        ("No work-life balance, I am burning out.", "NEGATIVE"),
        ("Management does not care about employees.", "NEGATIVE"),
        ("I hate coming to work every day.", "NEGATIVE")
    ]
    
    # 2. Setup Transformer (DistilBERT SST-2)
    print("Loading Transformer Pipeline...")
    start_load = time.time()
    try:
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        print(f"Transformer loaded in {time.time() - start_load:.2f}s")
    except Exception as e:
        print(f"Failed to load transformer: {e}")
        return

    # 3. Run Comparison
    vader_correct = 0
    trans_correct = 0
    vader_times = []
    trans_times = []
    
    print("\nRunning Inference...")
    for text, label in data:
        # VADER
        t0 = time.time()
        v_res = ml_service.analyze_sentiment(text)
        dt_v = (time.time() - t0) * 1000
        vader_times.append(dt_v)
        
        # VADER Logic: Compound > 0.05 is Positive
        v_pred = "POSITIVE" if v_res['compound'] >= 0.05 else "NEGATIVE"
        if v_pred == label:
            vader_correct += 1
            
        # Transformer
        t0 = time.time()
        t_res = classifier(text)[0] # [{'label': 'POSITIVE', 'score': 0.99}]
        dt_t = (time.time() - t0) * 1000
        trans_times.append(dt_t)
        
        t_pred = t_res['label']
        if t_pred == label:
            trans_correct += 1
            
    # 4. Report
    print("\n| Model | Accuracy (N=10) | Avg Latency | Speedup Factor |")
    print("| :--- | :--- | :--- | :--- |")
    
    v_acc = (vader_correct / len(data)) * 100
    t_acc = (trans_correct / len(data)) * 100
    v_lat = sum(vader_times) / len(vader_times)
    t_lat = sum(trans_times) / len(trans_times)
    speedup = t_lat / v_lat if v_lat > 0 else 0
    
    print(f"| VADER | {v_acc:.0f}% | {v_lat:.4f} ms | 1x (Baseline) |")
    print(f"| DistilBERT | {t_acc:.0f}% | {t_lat:.4f} ms | {speedup:.1f}x Slower |")

if __name__ == "__main__":
    verify_transformer_comparison()
