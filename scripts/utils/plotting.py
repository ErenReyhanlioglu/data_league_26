import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """
    Sets the global scientific style for all plots.
    """
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12

def plot_feature_importance(importance_df, title="Feature Importance", top_n=20):
    """
    Plots a horizontal bar chart of feature importances.
    """
    plt.figure(figsize=(10, top_n * 0.4))
    sns.barplot(
        x='importance', 
        y='feature', 
        data=importance_df.head(top_n), 
        hue='feature', 
        legend=False,
        palette='viridis'
    )
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

def plot_distribution_comparison(df1, df2, column, label1="Train", label2="Test"):
    """
    Compares the distribution of a feature between two datasets.
    Crucial for detecting distribution shifts.
    """
    plt.figure(figsize=(12, 5))
    sns.kdeplot(df1[column], label=label1, fill=True, alpha=0.5)
    sns.kdeplot(df2[column], label=label2, fill=True, alpha=0.5)
    plt.title(f"Distribution Comparison: {column}")
    plt.legend()
    plt.show()

def plot_target_correlation(df, feature_cols, target_col='label_noshow'):
    """
    Plots the correlation of features with the target variable.
    """
    correlations = df[feature_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
    correlations = correlations.drop(target_col) # Drop self-correlation
    
    plt.figure(figsize=(10, len(feature_cols) * 0.4))
    correlations.plot(kind='barh', color='skyblue')
    plt.title(f"Correlation with {target_col}")
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()

def plot_categorical_noshow_rate(df, cat_col, target_col='label_noshow', top_n=15):
    """
    Visualizes the No-Show rate across different categories.
    Example: Which 'specialty' has the highest no-show rate?
    """
    order = df.groupby(cat_col)[target_col].mean().sort_values(ascending=False).index[:top_n]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=cat_col, y=target_col, data=df, order=order, palette='magma', errorbar=None)
    plt.title(f"No-Show Rate by {cat_col} (Top {top_n})")
    plt.xticks(rotation=45)
    plt.ylabel("Mean No-Show Rate")
    plt.show()

def plot_bivariate_target_rate(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> None:
    """
    Plots the volume of observations and the mean target rate across bins of a specific feature.
    Corrects the x-axis sorting issue by maintaining mathematical interval order and resolves Line2D attribute errors.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        feature (str): The independent variable to analyze.
        target (str): The binary target variable (0 or 1).
        bins (int): Number of bins for continuous variables.
    """
    df_temp = df[[feature, target]].copy()
    
    # Binning continuous features while preserving the Interval categorical data type
    if pd.api.types.is_numeric_dtype(df_temp[feature]) and df_temp[feature].nunique() > bins:
        df_temp['bin'] = pd.qcut(df_temp[feature], q=bins, duplicates='drop')
    else:
        df_temp['bin'] = df_temp[feature]
        
    # Aggregate data (observed=True ensures we only group by existing categories)
    agg_df = df_temp.groupby('bin', observed=True).agg(
        volume=(target, 'count'),
        target_rate=(target, 'mean')
    ).reset_index()
    
    # Sort logically by the mathematical interval
    agg_df = agg_df.sort_values('bin')
    
    # Convert to string exclusively for visualization purposes on the x-axis
    agg_df['bin_str'] = agg_df['bin'].astype(str)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Bar plot for volume (Primary Y-axis)
    sns.barplot(data=agg_df, x='bin_str', y='volume', color='lightsteelblue', ax=ax1, alpha=0.8)
    ax1.set_xlabel(f'{feature} Bins (Mathematically Sorted)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Observation Volume', fontsize=12, color='slategrey')
    ax1.tick_params(axis='x', rotation=45)
    
    # Line plot for target rate (Secondary Y-axis) - Removed invalid 'group' parameter
    ax2 = ax1.twinx()
    sns.lineplot(data=agg_df, x='bin_str', y='target_rate', color='darkred', marker='o', 
                 linewidth=2.5, markersize=8, ax=ax2)
    ax2.set_ylabel('Mean Target Rate', fontsize=12, color='darkred')
    
    # Annotations for exact target rates using enumerate for robust spatial indexing
    for idx, row in enumerate(agg_df.itertuples()):
        ax2.annotate(f"{row.target_rate:.3f}", 
                     (idx, row.target_rate),
                     textcoords="offset points", 
                     xytext=(0,12), 
                     ha='center', 
                     fontsize=10,
                     fontweight='bold',
                     color='darkred')

    plt.title(f'Corrected Bivariate Analysis: {feature} vs {target}', fontsize=15, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Example execution call:
# plot_bivariate_target_rate(df=sub_train_finel_feature_38, feature='lead_time_hours', target='label_noshow')

def plot_clustered_correlation(df: pd.DataFrame, features_list: list) -> None:
    """
    Plots a hierarchically clustered heatmap of the correlation matrix for a given list of features.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        features_list (list): List of numerical features to include in the correlation matrix.
    """
    # Filter numerical columns
    num_df = df[features_list].select_dtypes(include=[np.number])
    
    if num_df.empty:
        print("Error: No numerical features provided for correlation analysis.")
        return
        
    # Calculate Spearman correlation to capture non-linear monotonic relationships
    corr_matrix = num_df.corr(method='spearman')
    
    # Fill NaN values (if any standard deviation is 0) to prevent clustering errors
    corr_matrix = corr_matrix.fillna(0)
    
    # Generate the clustered heatmap
    g = sns.clustermap(corr_matrix, 
                       method='ward', 
                       cmap='vlag', 
                       vmin=-1, vmax=1, 
                       figsize=(14, 14),
                       linewidths=0.5,
                       annot=False, # Set to True if fewer features
                       dendrogram_ratio=(0.1, 0.2),
                       cbar_pos=(0.02, 0.8, 0.03, 0.18))
    
    g.fig.suptitle('Hierarchical Clustered Spearman Correlation Matrix', fontsize=16, fontweight='bold', y=1.02)
    plt.show()

# Example execution call:
# plot_clustered_correlation(df=sub_train_finel_feature_38, features_list=finel_feature_38)

def plot_feature_target_distribution(df: pd.DataFrame, features: list, target: str) -> None:
    """
    Plots asymmetric violin plots to compare feature distributions across the binary target classes.
    Applies Winsorization (1st-99th percentile clipping) purely for visualization stability,
    preventing extreme outliers from distorting the Kernel Density Estimation (KDE) and Z-score scale.
    
    Args:
        df (pd.DataFrame): The input dataframe.
        features (list): List of feature names to plot.
        target (str): The binary target variable.
    """
    # Restrict to maximum 6 features at a time to maintain visual clarity
    features_to_plot = features[:6]
    
    # Standardize data strictly for visualization scale comparability using Z-score
    df_viz = df[features_to_plot + [target]].copy()
    
    for col in features_to_plot:
        if pd.api.types.is_numeric_dtype(df_viz[col]):
            # 1. Isolate the core 98% of the data (Winsorization) to fix KDE distortion
            lower_bound = df_viz[col].quantile(0.01)
            upper_bound = df_viz[col].quantile(0.99)
            df_viz[col] = df_viz[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 2. Standardize (Z-score) the clipped data
            mean_val = df_viz[col].mean()
            std_val = df_viz[col].std()
            
            if std_val > 0:
                df_viz[col] = (df_viz[col] - mean_val) / std_val
            else:
                df_viz[col] = 0

    # Melt dataframe for seaborn format
    df_melted = pd.melt(df_viz, id_vars=[target], value_vars=features_to_plot, 
                        var_name='Feature', value_name='Standardized Value')

    plt.figure(figsize=(14, 8))
    
    # Create the split violin plot with cut=0 to strictly limit KDE to the data range
    sns.violinplot(data=df_melted, x='Feature', y='Standardized Value', hue=target, 
                   split=True, inner='quartile', palette={0: 'skyblue', 1: 'salmon'},
                   linewidth=1.2, density_norm='width', cut=0)
    
    plt.title('Feature Distributions Across Target Classes (Winsorized & Standardized)', fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Z-Score (Clipped Range)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Force Y-axis limits to logical Z-score boundaries for visual consistency
    plt.ylim(-4, 4)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example execution call (from the batch loop provided previously):
# plot_feature_target_distribution(df=sub_train_pure, features=current_batch, target='label_noshow')