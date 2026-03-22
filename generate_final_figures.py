import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# Ensure the backend directory is in the path to import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.ml.prediction import ml_service

def generate_confusion_matrix():
    if not hasattr(ml_service, 'X_test') or ml_service.attrition_pipeline is None:
        print("ML Service not correctly initialized.")
        return

    # Use the real model predictions instead of predefined metrics
    probs = ml_service.attrition_pipeline.predict_proba(ml_service.X_test)[:, 1]
    sharp_probs = ml_service._sharpen(probs)
    y_pred = (sharp_probs >= ml_service.opt_threshold).astype(int)
    
    cm = confusion_matrix(ml_service.y_test, y_pred)
    acc = accuracy_score(ml_service.y_test, y_pred) * 100
    f1 = f1_score(ml_service.y_test, y_pred) * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Predicted: NO', 'Predicted: YES'],
                yticklabels=['Actual: NO', 'Actual: YES'],
                cbar=False, annot_kws={"size": 16})
    
    plt.title(f'Attrition Prediction: Hybrid Peak Ensemble\n(Accuracy: {acc:.2f}%, F1-Score: {f1:.2f}%)', fontsize=14, pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    output_path = 'fig1_confusion_matrix_final.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path} dynamically.")

def generate_performance_comparison():
    if not hasattr(ml_service, 'X_test') or ml_service.attrition_pipeline is None:
        return

    voting_clf = ml_service.attrition_pipeline.named_steps['classifier']
    preprocessor = ml_service.attrition_pipeline.named_steps['preprocessor']
    X_test_transformed = preprocessor.transform(ml_service.X_test)
    
    # VotingClassifier's estimators_ match the passed list: Random Forest, Logistic Regression, SVC, XGBoost
    model_names = ['Random Forest', 'Logistic Regression', 'SVM', 'XGBoost']
    
    accuracies = []
    f1s = []
    
    for clf in voting_clf.estimators_:
        y_pred = clf.predict(X_test_transformed)
        accuracies.append(accuracy_score(ml_service.y_test, y_pred))
        f1s.append(f1_score(ml_service.y_test, y_pred, zero_division=0))
        
    # Add Hybrid Ensemble
    probs = ml_service.attrition_pipeline.predict_proba(ml_service.X_test)[:, 1]
    sharp_probs = ml_service._sharpen(probs)
    y_pred_ens = (sharp_probs >= ml_service.opt_threshold).astype(int)
    
    model_names.append('Hybrid Ensemble')
    accuracies.append(accuracy_score(ml_service.y_test, y_pred_ens))
    f1s.append(f1_score(ml_service.y_test, y_pred_ens, zero_division=0))

    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#2ecc71')
    rects2 = ax.bar(x + width/2, f1s, width, label='F1-Score', color='#27ae60')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (Live Computed)', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.0)
    
    # Add data labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        horizontalalignment='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    
    output_path = 'fig2_performance_metrics_final.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path} dynamically.")

if __name__ == "__main__":
    generate_confusion_matrix()
    generate_performance_comparison()
