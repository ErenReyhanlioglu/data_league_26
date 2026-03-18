import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

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
            
    # Convert to DataFrame and sort from highest predictive power to lowest
    results_df = pd.DataFrame(auc_results)
    results_df = results_df.sort_values(by='Univariate_AUC', ascending=False).reset_index(drop=True)
    
    # Display the top and bottom features
    print("\n--- Top 10 Strongest Features ---")
    print(results_df.head(10).to_string(index=False))
    
    print("\n--- Top 5 Weakest Features ---")
    print(results_df.tail(5).to_string(index=False))
    
    return results_df

# Example Execution:
# meta_cols = ['appointment_id', 'patient_id', 'clinic_id', 'label_noshow', 'appointment_datetime', 'booking_datetime']
# pure_features_to_test = [col for col in sub_train_pure.columns if col not in meta_cols]
# auc_ranking = evaluate_univariate_auc(df=sub_train_pure, features=pure_features_to_test, target='label_noshow')