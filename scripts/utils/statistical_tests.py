import pandas as pd
import numpy as np

def drop_zero_variance(df):
    """
    Identifies features with zero variance (constants).
    Returns the reduced DataFrame and a detailed report of all features.
    """
    initial_cols = df.shape[1]
    nunique = df.nunique()
    
    # Create a detailed report for all columns
    report_df = pd.DataFrame({
        'Feature': nunique.index,
        'Unique_Values': nunique.values
    })
    
    # Flag features to be dropped
    report_df['Action'] = np.where(report_df['Unique_Values'] <= 1, 'Dropped', 'Kept')
    
    cols_to_drop = report_df[report_df['Action'] == 'Dropped']['Feature'].tolist()
    df_reduced = df.drop(columns=cols_to_drop)
    
    dropped_count = len(cols_to_drop)
    print(f"Variance Check: Dropped {dropped_count} zero-variance features.")
    
    return df_reduced, report_df.sort_values(by='Unique_Values')

def drop_high_correlation(df, threshold=0.90):
    """
    Identifies highly correlated numeric features.
    Returns the reduced DataFrame, a report of highly correlated pairs, and the drop list.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    pairs = []
    cols_to_drop = set()
    
    # Extract exact pairs and their correlation scores
    for col in upper.columns:
        high_corr_rows = upper.index[upper[col] > threshold].tolist()
        for row in high_corr_rows:
            corr_score = upper.loc[row, col]
            pairs.append({
                'Feature_1 (Kept)': row,
                'Feature_2 (Dropped)': col,
                'Correlation_Score': corr_score
            })
            cols_to_drop.add(col)
            
    # Create a detailed report of dropped pairs
    report_df = pd.DataFrame(pairs).sort_values(by='Correlation_Score', ascending=False)
    
    df_reduced = df.drop(columns=list(cols_to_drop))
    print(f"Correlation Check: Dropped {len(cols_to_drop)} features (threshold > {threshold}).")
    
    return df_reduced, report_df, list(cols_to_drop)

def calculate_iv_woe(df, target_col, feature_cols):
    """
    Calculates Information Value (IV) for given features.
    Returns a detailed dataframe classifying every feature's predictive power.
    """
    iv_dict = {}
    
    for col in feature_cols:
        if df[col].nunique() <= 1:
            continue
            
        temp_df = pd.DataFrame({'feature': df[col], 'target': df[target_col]})
        
        # Bin continuous variables robustly
        if temp_df['feature'].dtype.kind in 'bifc' and temp_df['feature'].nunique() > 10:
            try:
                temp_df['feature_bin'] = pd.qcut(temp_df['feature'], q=10, duplicates='drop')
            except ValueError:
                temp_df['feature_bin'] = temp_df['feature'].astype(str)
        else:
            temp_df['feature_bin'] = temp_df['feature']
            
        # Group by bins and calculate events/non-events
        stats = temp_df.groupby('feature_bin', observed=True)['target'].agg(['count', 'sum'])
        stats.columns = ['Total', 'Bad']
        stats['Good'] = stats['Total'] - stats['Bad']
        
        # Filter out bins with zero counts
        stats = stats[(stats['Good'] > 0) & (stats['Bad'] > 0)]
        if len(stats) == 0:
            continue
            
        stats['Dist_Good'] = stats['Good'] / stats['Good'].sum()
        stats['Dist_Bad'] = stats['Bad'] / stats['Bad'].sum()
        
        # Calculate WoE and IV
        stats['WoE'] = np.log(stats['Dist_Good'] / stats['Dist_Bad'])
        stats['IV'] = (stats['Dist_Good'] - stats['Dist_Bad']) * stats['WoE']
        
        iv_dict[col] = stats['IV'].sum()
        
    iv_df = pd.DataFrame(list(iv_dict.items()), columns=['Feature', 'IV']).sort_values(by='IV', ascending=False)
    
    # Add detailed classification status based on scientific thresholds
    conditions = [
        (iv_df['IV'] < 0.02),
        (iv_df['IV'] >= 0.02) & (iv_df['IV'] < 0.1),
        (iv_df['IV'] >= 0.1) & (iv_df['IV'] < 0.3),
        (iv_df['IV'] >= 0.3) & (iv_df['IV'] < 0.5),
        (iv_df['IV'] >= 0.5)
    ]
    choices = ['Useless (Drop)', 'Weak', 'Medium', 'Strong', 'Suspicious (Leakage?)']
    iv_df['Predictive_Power'] = np.select(conditions, choices, default='Unknown')
    
    strong_features = iv_df[iv_df['Predictive_Power'] != 'Useless (Drop)']['Feature'].tolist()
    
    print(f"IV Analysis: Kept {len(strong_features)} features, Dropped {len(iv_df) - len(strong_features)} features.")
    
    return iv_df, strong_features

def remove_collinear_features(df: pd.DataFrame, target_col: str, threshold: float = 0.85) -> list:
    """
    Identifies and mathematically eliminates collinear features to resolve multicollinearity.
    Retains the feature with the highest absolute correlation to the target variable.
    
    Args:
        df (pd.DataFrame): Dataset containing features and target.
        target_col (str): The name of the target variable.
        threshold (float): Pearson/Spearman correlation threshold for elimination.
        
    Returns:
        list: The final list of features after dropping redundant ones.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns:
        raise ValueError("Target column must be numeric for correlation calculation.")
        
    target_correlations = numeric_df.corrwith(numeric_df[target_col]).abs()
    
    feature_cols = [c for c in numeric_df.columns if c != target_col]
    corr_matrix = numeric_df[feature_cols].corr().abs()
    
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    features_to_drop = set()
    
    for column in upper_tri.columns:
        correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()
        
        for corr_feat in correlated_features:
            corr_feat_target_corr = target_correlations.get(corr_feat, 0)
            column_target_corr = target_correlations.get(column, 0)
            
            if corr_feat_target_corr > column_target_corr:
                features_to_drop.add(column)
            else:
                features_to_drop.add(corr_feat)
                
    final_features = [f for f in feature_cols if f not in features_to_drop]
    
    print(f"Initial independent feature count: {len(feature_cols)}")
    print(f"Features eliminated due to redundancy (|r| > {threshold}): {len(features_to_drop)}")
    print(f"Remaining pure feature count: {len(final_features)}\n")
    
    print("Eliminated Features:")
    for f in sorted(list(features_to_drop)):
        print(f" - {f}")
        
    return final_features
