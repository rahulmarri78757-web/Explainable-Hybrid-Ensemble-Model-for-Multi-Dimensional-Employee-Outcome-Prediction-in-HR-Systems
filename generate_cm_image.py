import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_styled_cm():
    # User-Requested Matrix Values (TN=343, FP=27, FN=18, TP=53)
    cm = np.array([[343, 27], [18, 53]])
    
    # User-requested labels
    labels = ['No Attrition', 'Attrtion'] # Matching the "Pred No" / "Actual No" style conceptually
    labels = ['Stayed', 'Left'] # Using the styled labels as requested in the previous turn, assumming they still want "Stayed/Left" if they refer to "before pic". 
    # Actually, looking at the request "| Pred No | Pred Yes |", let's stick to "Stayed" (No) and "Left" (Yes) as that is standard for this dataset and likely what the "before pic" had. 
    # Or I can use "No / Yes" to match the text table in the prompt.
    # Let's use "Stayed" / "Left" as it's cleaner, but I will check if the user wants specific labels. 
    # The prompt table says "Pred No", "Pred Yes".
    # Let's use "No" and "Yes" to be safe and match the table EXACTLY.
    labels = ['No', 'Yes']

    plt.figure(figsize=(8, 6))
    
    # Create heatmap with Colorbar (cbar=True)
    # Using 'Blues' cmap
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                     xticklabels=labels, yticklabels=labels,
                     annot_kws={"size": 16}) 
    
    plt.xlabel('Predicted Attrition', fontsize=14)
    plt.ylabel('Actual Attrition', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16) 
    
    output_path = r"C:\Users\ADMIN\.gemini\antigravity\brain\743f2dcb-a693-425b-b591-aec30e201aea\confusion_matrix_final.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Styled confusion matrix saved to: {output_path}")

if __name__ == "__main__":
    generate_styled_cm()
