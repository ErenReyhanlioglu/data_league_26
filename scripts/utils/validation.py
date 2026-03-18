import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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
        
        # Using a shallow Random Forest for quick distribution check
        model = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, probs))
        importances += model.feature_importances_ / 5
        
    return np.mean(auc_scores), pd.Series(importances, index=features).sort_values(ascending=False)