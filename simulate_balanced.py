import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def simulate_balanced_eval():
    # Targets from user
    # Acc 87.53, Prec 89.58, Rec 86.58
    
    # Let's assume a balanced test set of 142 samples (71 Yes, 71 No)
    # Total = 142
    # Pos = 71, Neg = 71
    
    print("--- SIMULATING BALANCED TEST SET (71 Yes / 71 No) ---")
    
    for tp in range(60, 65):
        # Recall Check: 61/71 = 85.9%, 62/71 = 87.3%
        rec_val = tp / 71
        
        # We need Accuracy ~87.53%
        # (TP + TN) / 142 = 0.8753
        # TP + TN = 124.3 -> let's try TN = 124 - TP
        tn = 124 - tp
        fp = 71 - tn
        fn = 71 - tp
        
        acc = (tp + tn) / 142
        prec_yes = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_yes = tp / (tp + fn)
        prec_no = tn / (tn + fn)
        rec_no = tn / (tn + fp)
        
        print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"  Acc: {acc:.4%}")
        print(f"  [YES] Prec: {prec_yes:.4%}, Rec: {rec_yes:.4%}")
        print(f"  [NO ] Prec: {prec_no:.4%}, Rec: {rec_no:.4%}")
        print("-" * 20)

if __name__ == "__main__":
    simulate_balanced_eval()
