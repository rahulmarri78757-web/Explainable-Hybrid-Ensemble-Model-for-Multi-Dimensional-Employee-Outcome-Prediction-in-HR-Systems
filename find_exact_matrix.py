import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def find_exact_matrix(target_acc, target_w_prec, target_w_rec, n_total=441, n_pos=71):
    print(f"Searching for matrix: Acc={target_acc}, W_Prec={target_w_prec}, W_Rec={target_w_rec}")
    
    n_neg = n_total - n_pos
    best_error = float('inf')
    best_cm = None
    
    # Accuracy is (TP + TN) / N
    # So TP + TN must be around target_acc * N
    correct_count = int(round(target_acc * n_total))
    print(f"Target Correct Count: {correct_count}")
    
    for tp in range(0, n_pos + 1):
        tn = correct_count - tp
        if tn < 0 or tn > n_neg: continue
        
        fp = n_neg - tn
        fn = n_pos - tp
        
        y_true = [0]*n_neg + [1]*n_pos
        y_pred = [0]*tn + [1]*fp + [0]*fn + [1]*tp
        
        acc = accuracy_score(y_true, y_pred)
        w_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        w_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        w_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        error = abs(acc*100 - target_acc) + abs(w_prec*100 - target_w_prec) + abs(w_rec*100 - target_w_rec)
        
        if error < best_error:
            best_error = error
            best_cm = {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'acc': acc*100, 'w_prec': w_prec*100, 'w_rec': w_rec*100, 'w_f1': w_f1*100
            }
            if error < 0.1:
                break
                
    if best_cm:
        print("\nBest Match Found:")
        for k, v in best_cm.items():
            print(f"{k}: {v:.4f}")
            
        # Class-wise
        p0 = best_cm['tn'] / (best_cm['tn'] + best_cm['fn'])
        r0 = best_cm['tn'] / (best_cm['tn'] + best_cm['fp'])
        p1 = best_cm['tp'] / (best_cm['tp'] + best_cm['fp']) if (best_cm['tp'] + best_cm['fp']) > 0 else 0
        r1 = best_cm['tp'] / (best_cm['tp'] + best_cm['fn'])
        print(f"\nClass 0 (No): P={p0:.2%}, R={r0:.2%}")
        print(f"Class 1 (Yes): P={p1:.2%}, R={r1:.2%}")

if __name__ == "__main__":
    # The user's numbers
    find_exact_matrix(87.53, 89.58, 86.58)
