
import warnings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import random
import numpy as np



# Sample Data Pool (Positive & Negative HR feedback)
POSITIVE_SAMPLES = [
    "I love working here, the environment is great.",
    "My manager is very supportive and helps me grow.",
    "Great work-life balance and good benefits.",
    "I feel valued and appreciated by my team.",
    "The company culture is amazing and very inclusive.",
    "Excellent opportunities for career development.",
    "I am very satisfied with my current role.",
    "The leadership team is transparent and honest.",
    "I enjoy the flexibility this job offers.",
    "The office atmosphere is very positive and energetic.",
    "I'm proud to be part of this organization.",
    "The projects are challenging and rewarding.",
    "Good salary and compensation package.",
    "Supportive colleagues make work enjoyable.",
    "I see a long-term future for myself here."
]

NEGATIVE_SAMPLES = [
    "I am frustrated with the lack of communication.",
    "My workload is unmanageable and causing stress.",
    "I don't feel supported by management.",
    "The pay is too low for the amount of work.",
    "There is no room for growth or advancement.",
    "I am considering leaving due to poor management.",
    "The company culture is toxic and competitive.",
    "I feel undervalued and ignored.",
    "Work-life balance is non-existent here.",
    "The benefits package is very poor.",
    "Micromanagement is a huge issue in this team.",
    "I am burnt out and need a break.",
    "The management does not listen to employee feedback.",
    "I regret joining this company.",
    "There is a lot of favoritism and bias."
]

def calibrate_sentiment():
    print("Evaluating Sentiment Validation Set...")
    analyzer = SentimentIntensityAnalyzer()
    
    texts = POSITIVE_SAMPLES + NEGATIVE_SAMPLES
    labels = [1]*len(POSITIVE_SAMPLES) + [0]*len(NEGATIVE_SAMPLES)
    
    preds = []
    for text in texts:
        score = analyzer.polarity_scores(text)['compound']
        preds.append(1 if score >= 0.05 else 0)
        
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    print(f"True Accuracy: {acc:.2%}")
    print(f"True F1-Score: {f1:.2%}")
    print(f"True Precision: {precision_score(labels, preds):.2%}")
    print(f"True Recall: {recall_score(labels, preds):.2%}")

if __name__ == "__main__":
    calibrate_sentiment()
