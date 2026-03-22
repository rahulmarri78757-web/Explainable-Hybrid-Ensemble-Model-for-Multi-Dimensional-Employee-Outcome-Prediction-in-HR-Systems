import numpy as np

def verify_user_table(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    n_no = tn + fp
    n_yes = fn + tp
    
    acc = (tp + tn) / total
    
    prec_no = tn / (tn + fn)
    rec_no = tn / (tn + fp)
    f1_no = 2 * (prec_no * rec_no) / (prec_no + rec_no)
    
    prec_yes = tp / (tp + fp)
    rec_yes = tp / (tp + fn)
    f1_yes = 2 * (prec_yes * rec_yes) / (prec_yes + rec_yes)
    
    w_prec = (n_no * prec_no + n_yes * prec_yes) / total
    w_rec = (n_no * rec_no + n_yes * rec_yes) / total
    w_f1 = (n_no * f1_no + n_yes * f1_yes) / total
    
    print(f"--- MATHEMATICAL AUDIT ---")
    print(f"Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp} | Total={total}")
    print(f"\nClass-Wise Results:")
    print(f"No Attrition: P={prec_no:.4%}, R={rec_no:.4%}, F1={f1_no:.4%}")
    print(f"Attrition   : P={prec_yes:.4%}, R={rec_yes:.4%}, F1={f1_yes:.4%}")
    
    print(f"\nMain Table (Weighted Averages):")
    print(f"Accuracy : {acc:.4%}")
    print(f"Precision: {w_prec:.4%}")
    print(f"Recall   : {w_rec:.4%}")
    print(f"F1       : {w_f1:.4%}")

if __name__ == "__main__":
    # Based on the user's numbers: R1=74.65%, P1=66.25% (TP ~53)
    # R0=92.70% (TN ~343)
    verify_user_table(tn=343, fp=27, fn=18, tp=53)
