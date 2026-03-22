import numpy as np
import pandas as pd

def check_mathematical_consistency(target_acc, target_prec, target_rec, n_total=441, n_pos=71):
    print(f"Target: Acc={target_acc:.4f}, Prec={target_prec:.4f}, Rec={target_rec:.4f}")
    print(f"Total N={n_total}, Pos={n_pos}")
    
    best_error = float('inf')
    best_config = None
    
    # Iterate through all possible TP counts
    for tp in range(0, n_pos + 1):
        fn = n_pos - tp
        rec = tp / n_pos if n_pos > 0 else 0
        
        # Calculate needed FP for target precision
        # Prec = TP / (TP + FP) -> FP = TP(1-Prec)/Prec
        needed_fp = tp * (1 - target_prec) / target_prec if target_prec > 0 else n_total
        
        # We must have TN + FP = 370
        for fp in range(max(0, int(needed_fp)-5), min(n_total - n_pos, int(needed_fp)+5) + 1):
            tn = (n_total - n_pos) - fp
            
            acc = (tp + tn) / n_total
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            error = abs(acc - target_acc) + abs(prec - target_prec) + abs(rec - target_rec)
            
            if error < best_error:
                best_error = error
                best_config = {
                    'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                    'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1
                }
                
    print("\nClosest Possible Fit:")
    print(best_config)
    print(f"Error: {best_error:.4f}")

if __name__ == "__main__":
    # User's Attrition Table Targets
    check_mathematical_consistency(0.8753, 0.8958, 0.8658)
    
    print("\n" + "="*40)
    print("Testing if these are WEIGHTED AVERAGES (weighted by class support)")
    # Acc is usually fixed. Weighted Prec = (prec0 * n0 + prec1 * n1) / total
    # This is more complex to search.
