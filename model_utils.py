# model_utils.py - Model training and evaluation utilities
# ========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import joblib
import shap
import os


def train_model(X_train, y_train, model_params):
    """Train Random Forest model"""
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    print("Model training complete")
    return model


def evaluate_model(model, X_test, y_test, save_path='figures/'):
    """Comprehensive model evaluation"""
    os.makedirs(save_path, exist_ok=True)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\nModel Performance:")
    print(f"  AUC-ROC: {auc_score:.3f}")
    print(f"  Average Precision: {ap_score:.3f}")

    # Handle both string and integer keys for fraud class
    fraud_key = '1' if '1' in report else (1 if 1 in report else None)
    if fraud_key:
        print(f"  Fraud Precision: {report[fraud_key]['precision']:.3f}")
        print(f"  Fraud Recall: {report[fraud_key]['recall']:.3f}")
        print(f"  Fraud F1-Score: {report[fraud_key]['f1-score']:.3f}")
    else:
        print("  Warning: No fraud cases found in test set!")

    # Create evaluation plots
    create_evaluation_plots(y_test, y_pred, y_pred_proba, cm, save_path)

    return {
        'classification_report': report,
        'auc_score': auc_score,
        'ap_score': ap_score,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def create_evaluation_plots(y_test, y_pred, y_pred_proba, cm, save_path):
    """Create model evaluation visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    axes[1, 0].plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Probability Distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Normal', density=True)
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraud', density=True)
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].set_xlabel('Fraud Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{save_path}model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_threshold_impact(y_test, y_pred_proba, save_path='figures/'):
    """Analyze impact of different classification thresholds"""
    os.makedirs(save_path, exist_ok=True)

    thresholds = np.arange(0.1, 1.0, 0.05)
    metrics = {'threshold': [], 'precision': [], 'recall': [], 'f1': []}

    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        tp = np.sum((y_test == 1) & (y_pred_thresh == 1))
        fp = np.sum((y_test == 0) & (y_pred_thresh == 1))
        fn = np.sum((y_test == 1) & (y_pred_thresh == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics['threshold'].append(threshold)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)

    # Plot threshold analysis
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['threshold'], metrics['precision'], label='Precision', linewidth=2)
    plt.plot(metrics['threshold'], metrics['recall'], label='Recall', linewidth=2)
    plt.plot(metrics['threshold'], metrics['f1'], label='F1-Score', linewidth=2, linestyle='--')
    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Performance Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return metrics


def save_model(model, scaler, feature_names, model_path):
    """Save model and preprocessing artifacts"""
    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'RandomForest'
    }
    joblib.dump(artifacts, model_path)
    print(f"\nModel saved to {model_path}")


def perform_shap_analysis(model, X_test, feature_names, save_path='figures/'):
    """Perform SHAP analysis for explainability"""
    print("  Performing SHAP analysis...")
    os.makedirs(save_path, exist_ok=True)

    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{save_path}shap_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Calculate importance safely
        shap_importance = np.abs(shap_values).mean(axis=0)

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(shap_importance)],
            'importance': shap_importance[:len(feature_names)]
        }).sort_values('importance', ascending=False)

        return importance_df

    except Exception as e:
        print(f"  Warning: SHAP analysis failed: {e}")
        # Return empty dataframe on failure
        return pd.DataFrame({'feature': feature_names, 'importance': [0.1] * len(feature_names)})