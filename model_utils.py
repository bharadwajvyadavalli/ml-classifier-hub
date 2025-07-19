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


def calculate_optimal_threshold(y_test, y_pred_proba):
    """Calculate optimal threshold using Youden's J statistic"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Youden's J statistic (Sensitivity + Specificity - 1)
    # J = TPR - FPR = TPR - (1 - TNR) = TPR + TNR - 1
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_j = j_scores[optimal_idx]
    
    return {
        'optimal_threshold': optimal_threshold,
        'optimal_tpr': optimal_tpr,
        'optimal_fpr': optimal_fpr,
        'optimal_j_score': optimal_j,
        'fpr_curve': fpr,
        'tpr_curve': tpr,
        'thresholds': thresholds
    }


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


def create_roc_analysis(y_test, y_pred_proba, save_path):
    """Create simple ROC curve with optimal J-statistic point"""
    # Calculate optimal threshold
    optimal_data = calculate_optimal_threshold(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create simple single plot
    plt.figure(figsize=(8, 6))
    
    # Plot ROC curve
    plt.plot(optimal_data['fpr_curve'], optimal_data['tpr_curve'], linewidth=2, 
            label=f'ROC Curve (AUC = {auc:.3f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Mark optimal J-statistic point using single values
    optimal_fpr = optimal_data['optimal_fpr']  # This is the single optimal FPR value
    optimal_tpr = optimal_data['optimal_tpr']  # This is the single optimal TPR value
    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10, 
            label=f"Optimal J-statistic\nThreshold = {optimal_data['optimal_threshold']:.3f}\nTPR = {optimal_tpr:.3f}, FPR = {optimal_fpr:.3f}\nJ = {optimal_data['optimal_j_score']:.3f}")
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Optimal J-statistic Point')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}roc_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print optimal threshold information
    print(f"\nOptimal J-statistic Analysis:")
    print(f"  Threshold: {optimal_data['optimal_threshold']:.3f}")
    print(f"  TPR: {optimal_tpr:.3f}")
    print(f"  FPR: {optimal_fpr:.3f}")
    print(f"  J-statistic: {optimal_data['optimal_j_score']:.3f}")
    
    return optimal_data


def create_evaluation_plots(y_test, y_pred, y_pred_proba, cm, save_path):
    """Create ROC-AUC and PR-AUC curves with J-statistic highlighted"""
    # Create 1x2 subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calculate ROC curve and J-statistic
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Calculate J-statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_j = j_scores[optimal_idx]
    
    # Plot ROC curve with J-statistic highlighted
    axes[0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})', color='blue')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    # Mark optimal J-statistic point with clear highlighting
    axes[0].plot(optimal_fpr, optimal_tpr, 'ro', markersize=12, 
                label=f"J-statistic = {optimal_j:.3f}\nThreshold = {optimal_threshold:.3f}")
    
    # Add annotation for J-statistic
    axes[0].annotate(f'J = {optimal_j:.3f}', 
                    xy=(optimal_fpr, optimal_tpr), 
                    xytext=(optimal_fpr + 0.1, optimal_tpr - 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, fontweight='bold', color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC-AUC Curve with J-statistic')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Calculate and plot PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    
    axes[1].plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {ap:.3f})', color='green')
    
    # Add baseline for PR curve (proportion of positive class)
    baseline = y_test.mean()
    axes[1].axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Baseline = {baseline:.3f}')
    
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print J-statistic information
    print(f"\nJ-statistic Analysis:")
    print(f"  J-statistic: {optimal_j:.3f}")
    print(f"  Threshold: {optimal_threshold:.3f}")
    print(f"  TPR: {optimal_tpr:.3f}")
    print(f"  FPR: {optimal_fpr:.3f}")


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


def save_model(model, scaler, feature_names, filepath):
    """Save trained model with metadata"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'RandomForest'
    }
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")


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