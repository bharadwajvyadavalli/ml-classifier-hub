# data_utils.py - Data loading and preprocessing utilities
# ========================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(filepath):
    """Load credit card dataset"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud cases: {df['Class'].sum():,} ({df['Class'].mean() * 100:.2f}%)")
    return df


def analyze_data(df, save_path='figures/'):
    """Perform exploratory data analysis"""
    os.makedirs(save_path, exist_ok=True)

    # Basic statistics
    stats = {
        'total_transactions': len(df),
        'fraud_count': df['Class'].sum(),
        'fraud_rate': df['Class'].mean() * 100,
        'avg_normal_amount': df[df['Class'] == 0]['Amount'].mean(),
        'avg_fraud_amount': df[df['Class'] == 1]['Amount'].mean()
    }

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Class distribution
    class_counts = df['Class'].value_counts()
    axes[0, 0].bar(['Normal', 'Fraud'], class_counts.values, color=['green', 'red'])
    axes[0, 0].set_title('Transaction Distribution')
    axes[0, 0].set_ylabel('Count')

    # 2. Amount distribution by class
    df[df['Class'] == 0]['Amount'].hist(bins=50, ax=axes[0, 1], alpha=0.7, label='Normal', density=True)
    df[df['Class'] == 1]['Amount'].hist(bins=50, ax=axes[0, 1], alpha=0.7, label='Fraud', density=True)
    axes[0, 1].set_title('Transaction Amount Distribution')
    axes[0, 1].set_xlabel('Amount')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 500)

    # 3. Feature correlations with fraud
    correlations = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)[:10]
    correlations.plot(kind='barh', ax=axes[1, 0], color='blue')
    axes[1, 0].set_title('Top 10 Features Correlated with Fraud')
    axes[1, 0].set_xlabel('Absolute Correlation')

    # 4. Time distribution
    df['Hour'] = (df['Time'] / 3600) % 24
    hourly_fraud_rate = df.groupby(df['Hour'].astype(int))['Class'].mean()
    axes[1, 1].plot(hourly_fraud_rate.index, hourly_fraud_rate.values * 100, marker='o')
    axes[1, 1].set_title('Fraud Rate by Hour of Day')
    axes[1, 1].set_xlabel('Hour')
    axes[1, 1].set_ylabel('Fraud Rate (%)')
    axes[1, 1].set_xticks(range(0, 24, 4))

    plt.tight_layout()
    plt.savefig(f'{save_path}eda_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return stats


def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare data for modeling"""
    # Separate features and target
    X = df.drop(['Class'], axis=1)
    y = df['Class']

    # Remove Hour column if it exists
    if 'Hour' in X.columns:
        X = X.drop('Hour', axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nData split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    print(f"  Features: {X.shape[1]}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def get_feature_importance(model, feature_names, top_n=15):
    """Extract and rank feature importances"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    return [(feature_names[i], importances[i]) for i in indices]


def calculate_vif(X_df, save_path='figures/'):
    """Calculate VIF for multicollinearity detection"""
    os.makedirs(save_path, exist_ok=True)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i)
                       for i in range(len(X_df.columns))]
    vif_sorted = vif_data.sort_values('VIF', ascending=False)

    # Create VIF visualization
    plt.figure(figsize=(10, 6))
    top_features = vif_sorted.head(15)
    colors = ['red' if vif > 10 else 'green' for vif in top_features['VIF']]

    plt.barh(range(len(top_features)), top_features['VIF'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Variance Inflation Factor (VIF)')
    plt.title('Top 15 Features by VIF Score')
    plt.axvline(x=10, color='red', linestyle='--', label='VIF = 10 (threshold)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}vif_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return vif_sorted


def perform_rfe(X_train, y_train, n_features=15):
    """Perform Recursive Feature Elimination"""
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    rfe = RFE(lr, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)

    # Print RFE ranking
    ranking = sorted(zip(range(len(rfe.ranking_)), rfe.ranking_), key=lambda x: x[1])
    print(f"  Top {n_features} features by RFE ranking")

    return rfe


def apply_smote(X_train, y_train):
    """Apply SMOTE for handling imbalance"""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"  SMOTE applied: {len(X_train)} → {len(X_resampled)} samples")
    return X_resampled, y_resampled