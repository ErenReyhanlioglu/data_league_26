import pandas as pd
import numpy as np
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    precision_recall_curve, 
    f1_score,
    classification_report
)
from sklearn.calibration import calibration_curve

def evaluate_univariate_auc(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """
    Calculates the Univariate ROC-AUC score for each feature to determine 
    its isolated predictive power against the binary target.
    
    Args:
        df (pd.DataFrame): The dataset containing features and target.
        features (list): List of feature names to evaluate.
        target (str): The binary target variable.
        
    Returns:
        pd.DataFrame: A dataframe with features ranked by their AUC scores.
    """
    auc_results = []
    
    # Drop rows where target is null, just in case
    valid_df = df.dropna(subset=[target])
    y_true = valid_df[target]
    
    print(f"Executing Univariate AUC evaluation for {len(features)} features...")
    
    for feature in features:
        # Skip features that have single unique value or are entirely null
        if valid_df[feature].nunique() <= 1 or valid_df[feature].isnull().all():
            print(f"Skipping {feature} due to zero variance or all nulls.")
            continue
            
        # Handle potential missing values in the feature by filling with median
        x_feature = valid_df[feature].fillna(valid_df[feature].median())
        
        try:
            # Calculate AUC. If the feature is negatively correlated with the target, 
            # the AUC might be < 0.5. We invert it (1 - AUC) to measure absolute predictive power.
            auc = roc_auc_score(y_true, x_feature)
            if auc < 0.5:
                auc = 1.0 - auc
                
            auc_results.append({'Feature': feature, 'Univariate_AUC': auc})
        except Exception as e:
            print(f"Error calculating AUC for {feature}: {e}")
            
    results_df = pd.DataFrame(auc_results)
    results_df = results_df.sort_values(by='Univariate_AUC', ascending=False).reset_index(drop=True)
    
    print("\n--- Top 10 Strongest Features ---")
    print(results_df.head(10).to_string(index=False))
    
    print("\n--- Top 5 Weakest Features ---")
    print(results_df.tail(5).to_string(index=False))
    
    return results_df

def calculate_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes a comprehensive suite of metrics for binary classification.
    Focuses on PR-AUC (Average Precision) as the primary competition metric.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'pr_auc': average_precision_score(y_true, y_prob),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'f1_score': f1_score(y_true, y_pred),
        'precision': classification_report(y_true, y_pred, output_dict=True)['1']['precision'],
        'recall': classification_report(y_true, y_pred, output_dict=True)['1']['recall']
    }
    return metrics

def get_pr_curve_data(y_true, y_prob):
    """Returns precision and recall values for plotting PR Curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return precision, recall, thresholds

def calculate_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Approximates the Expected Calibration Error (ECE).
    Measures the gap between predicted probability and actual frequency.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ece = np.mean(np.abs(prob_true - prob_pred))
    return ece