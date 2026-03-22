import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def find_exact_matrix(target_acc, target_w_prec, target_w_rec, n_total=441, n_pos=71):
    print(f"Searching for matrix: Acc={target_acc}, Prec={target_w_prec}, Rec={target_w_rec}")
    
    n_neg = n_total - n_pos
    best_error = float('inf')
    best_cm = None
    
    # Accuracy is (TP + TN) / N
    correct_count = int(round((target_acc / 100) * n_total))
    print(f"Target Correct Count: {correct_count}")
    
    for tp in range(0, n_pos + 1):
        tn = correct_count - tp
        if tn < 0 or tn > n_neg: continue
        
        fp = n_neg - tn
        fn = n_pos - tp
        
        y_true = np.array([0]*n_neg + [1]*n_pos)
        y_pred = np.array([0]*tn + [1]*fp + [0]*fn + [1]*tp)
        
        acc = accuracy_score(y_true, y_pred)
        
        # Test WEIGHTED
        w_prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        w_rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Test MACRO
        m_prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        m_rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        error_w = abs(acc*100 - target_acc) + abs(w_prec*100 - target_w_prec) + abs(w_rec*100 - target_w_rec)
        error_m = abs(acc*100 - target_acc) + abs(m_prec*100 - target_w_prec) + abs(m_rec*100 - target_w_rec)
        
        if error_w < best_error:
            best_error = error_w
            best_cm = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'acc': acc*100, 'prec': w_prec*100, 'rec': w_rec*100, 'type': 'Weighted'}
        
        if error_m < best_error:
            best_error = error_m
            best_cm = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'acc': acc*100, 'prec': m_prec*100, 'rec': m_rec*100, 'type': 'Macro'}

    if best_cm:
        print(f"\nBest Match Found ({best_cm['type']}):")
        for k, v in best_cm.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
            
        # Class-wise
        p0 = best_cm['tn'] / (best_cm['tn'] + best_cm['fn']) if (best_cm['tn'] + best_cm['fn']) > 0 else 0
        r0 = best_cm['tn'] / 370
        p1 = best_cm['tp'] / (best_cm['tp'] + best_cm['fp']) if (best_cm['tp'] + best_cm['fp']) > 0 else 0
        r1 = best_cm['tp'] / 71
        print(f"\nClass-wise Analysis (N_No=370, N_Yes=71):")
        print(f"No Attrition: P={p0:.2%}, R={r0:.2%}")
        print(f"Attrition:    P={p1:.2%}, R={r1:.2%}")

if __name__ == "__main__":
    find_exact_matrix(87.53, 89.58, 86.58)
