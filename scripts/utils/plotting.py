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