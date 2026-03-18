import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class ValidationEngine:
    """
    Core validation logic for temporal stability and distribution shift analysis.
    Designed for high-stakes competition environments and Fintech risk modeling.
    """
    
    @staticmethod
    def get_expanding_window_splits(df, date_col, start_month=7, end_month=10):
        """
        Generates chronological indices for Expanding Window Cross-Validation.
        Example: Fold 1 (Jan-Jun -> Jul), Fold 2 (Jan-Jul -> Aug), etc.
        
        Args:
            df (pd.DataFrame): Input dataset.
            date_col (str): Column name containing appointment month or datetime.
            start_month (int): The first month to be used as validation.
            end_month (int): The last month to be used as validation.
            
        Returns:
            list: A list of tuples (train_index, val_index)
        """
        splits = []
        # Ensure date_col is extracted as month if it's a datetime
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            months = df[date_col].dt.month
        else:
            months = df[date_col]
            
        for m in range(start_month, end_month + 1):
            train_idx = df[months < m].index.tolist()
            val_idx = df[months == m].index.tolist()
            splits.append((train_idx, val_idx))
            
        return splits

    @staticmethod
    def run_adversarial_validation(train_df, test_df, features):
        """
        Detects distribution shift between training and test sets.
        If AUC > 0.70, it indicates a significant drift that needs to be addressed.
        """
        X_train = train_df[features].copy()
        X_test = test_df[features].copy()
        
        X_train['is_test'] = 0
        X_test['is_test'] = 1
        
        data = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        X = data.drop(columns=['is_test'])
        y = data['is_test']
        
        # Using a fast Random Forest to distinguish sets
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
        
        mean_auc = np.mean(scores)
        print(f"--- Adversarial Validation Report ---")
        print(f"Mean AUC: {mean_auc:.4f}")
        
        if mean_auc > 0.70:
            print("WARNING: Significant distribution shift detected between Train and Test.")
        else:
            print("SUCCESS: Train and Test distributions are statistically similar.")
            
        return mean_auc

    @staticmethod
    def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
        """
        Calculates reliability curve data to assess probability calibration.
        Essential for checking if predicted 20% means 20% actual frequency.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        # Expected Calibration Error (ECE) approximation
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        return {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'ece': ece
        }

    @staticmethod
    def generate_stability_report(cv_results):
        """
        Aggregates PR-AUC results across temporal folds to measure variance.
        
        Args:
            cv_results (list): List of scores from each expanding window fold.
        """
        mean_score = np.mean(cv_results)
        std_score = np.std(cv_results)
        cv_range = np.max(cv_results) - np.min(cv_results)
        
        report = {
            'mean_pr_auc': mean_score,
            'std_dev': std_score,
            'range': cv_range,
            'stability_index': 1 - (std_score / mean_score) if mean_score > 0 else 0
        }
        
        print(f"\n--- Model Stability Report ---")
        print(f"Mean PR-AUC: {mean_score:.4f}")
        print(f"Std Deviation: {std_score:.4f}")
        print(f"Stability Index: {report['stability_index']:.4f}")
        
        return report

def split_by_time(df, split_date, date_col='appointment_datetime'):
    """
    Splits the dataframe chronologically based on a threshold date.
    Required for local validation strategy (e.g., separating October 2025).
    """
    train_idx = df[df[date_col] < split_date].index
    val_idx = df[df[date_col] >= split_date].index
    return df.loc[train_idx].copy(), df.loc[val_idx].copy()

def prepare_adversarial_data(train_df, test_df, features):
    """
    Merges train and test sets to create an 'is_test' target for distribution shift detection.
    """
    train_adv = train_df[features].copy()
    test_adv = test_df[features].copy()
    
    train_adv['is_test'] = 0
    test_adv['is_test'] = 1
    
    return pd.concat([train_adv, test_adv], axis=0).reset_index(drop=True)

def run_adversarial_validation(df, features, target='is_test'):
    """
    Trains a Random Forest to identify if train and test sets are statistically distinguishable.
    Returns the mean AUC score and feature importances.
    """
    X = df[features]
    y = df[target]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    importances = np.zeros(len(features))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, probs))
        importances += model.feature_importances_ / 5
        
    return np.mean(auc_scores), pd.Series(importances, index=features).sort_values(ascending=False)