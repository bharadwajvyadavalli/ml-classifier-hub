# train_model.py - Main training pipeline
# =======================================

import os
import pandas as pd
import config
from data_utils import (load_data, analyze_data, prepare_data, get_feature_importance,
                        calculate_vif, perform_rfe, apply_smote)
from model_utils import (train_model, evaluate_model, analyze_threshold_impact,
                         save_model, perform_shap_analysis)
from report_generator import generate_html_report


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Credit Card Fraud Detection - Training Pipeline")
    print("=" * 60)

    # 1. Load and analyze data
    print("\n1. Loading data...")
    df = load_data(config.DATA_PATH)

    print("\n2. Analyzing data...")
    data_stats = analyze_data(df, config.FIGURES_DIR)

    # 2. Prepare data
    print("\n3. Preparing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        df, config.TEST_SIZE, config.RANDOM_STATE
    )

    # 3. Calculate VIF
    print("\n4. Calculating VIF for multicollinearity...")
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    vif_results = calculate_vif(X_train_df, config.FIGURES_DIR)
    print(f"  Top 5 VIF scores:\n{vif_results.head()}")

    # Remove high VIF features
    high_vif_features = vif_results[vif_results['VIF'] > 10]['Feature'].tolist()
    if high_vif_features:
        print(f"  Removing {len(high_vif_features)} features with VIF > 10")
        feature_names = [f for f in feature_names if f not in high_vif_features]
        X_train = X_train_df[feature_names].values
        X_test = pd.DataFrame(X_test, columns=vif_results['Feature'])[feature_names].values

    # 4. Perform RFE
    print("\n5. Performing Recursive Feature Elimination...")
    rfe = perform_rfe(X_train, y_train, config.TOP_FEATURES)
    selected_features = [feature_names[i] for i, selected in enumerate(rfe.support_) if selected]
    print(f"  Selected {len(selected_features)} features: {selected_features[:5]}...")

    # Apply RFE selection
    X_train = X_train[:, rfe.support_]
    X_test = X_test[:, rfe.support_]

    # 5. Apply SMOTE
    print("\n6. Applying SMOTE for class balancing...")
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # 6. Train model
    print("\n7. Training model...")
    model = train_model(X_train_balanced, y_train_balanced, config.MODEL_PARAMS)

    # 7. Evaluate model
    print("\n8. Evaluating model...")
    results = evaluate_model(model, X_test, y_test, config.FIGURES_DIR)

    # 8. SHAP Analysis
    print("\n9. Performing SHAP analysis...")
    shap_importance = perform_shap_analysis(model, X_test, selected_features, config.FIGURES_DIR)

    # 9. Analyze thresholds
    print("\n10. Analyzing classification thresholds...")
    threshold_metrics = analyze_threshold_impact(
        y_test, results['probabilities'], config.FIGURES_DIR
    )

    # 10. Get feature importance
    print("\n11. Extracting feature importance...")
    feature_importance = get_feature_importance(model, selected_features, len(selected_features))

    # 11. Save model
    print("\n12. Saving model...")
    save_model(model, scaler, selected_features, config.MODEL_PATH)

    # 12. Generate report with all analyses
    print("\n13. Generating HTML report...")
    analysis_results = {
        'vif_results': vif_results.to_dict('records'),
        'selected_features': selected_features,
        'shap_importance': shap_importance.to_dict('records'),
        'smote_applied': True
    }

    generate_html_report(
        data_stats, results, feature_importance,
        config.FIGURES_DIR, config.REPORT_PATH,
        analysis_results
    )

    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print(f"📊 Model saved to: {config.MODEL_PATH}")
    print(f"📄 Report saved to: {config.REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()