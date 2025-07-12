# report_generator.py - Generate HTML report from ML results
# ==========================================================

import os
import base64
from datetime import datetime


def encode_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def generate_html_report(data_stats, model_results, feature_importance,
                         figures_dir='figures/', output_path='fraud_report.html',
                         analysis_results=None):
    """Generate comprehensive HTML report"""

    # Extract metrics
    report = model_results['classification_report']
    cm = model_results['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()

    # Handle both string and integer keys for fraud class
    fraud_key = '1' if '1' in report else (1 if 1 in report else None)
    if fraud_key:
        precision = report[fraud_key]['precision']
        recall = report[fraud_key]['recall']
        f1_score = report[fraud_key]['f1-score']
    else:
        precision = 0
        recall = 0
        f1_score = 0

    # Calculate business metrics
    fraud_caught = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Estimated financial impact
    avg_fraud_amount = data_stats['avg_fraud_amount']
    fraud_prevented = tp * avg_fraud_amount
    false_alarm_cost = fp * 10  # Assumed cost per false positive
    net_benefit = fraud_prevented - false_alarm_cost

    # Load images
    eda_img = encode_image(f'{figures_dir}eda_analysis.png')
    eval_img = encode_image(f'{figures_dir}model_evaluation.png')
    threshold_img = encode_image(f'{figures_dir}threshold_analysis.png') if os.path.exists(
        f'{figures_dir}threshold_analysis.png') else None
    shap_img = encode_image(f'{figures_dir}shap_importance.png') if os.path.exists(
        f'{figures_dir}shap_importance.png') else None
    vif_img = encode_image(f'{figures_dir}vif_analysis.png') if os.path.exists(
        f'{figures_dir}vif_analysis.png') else None

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Fraud Detection Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
            margin: -20px -20px 30px -20px;
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        h2 {{
            color: #667eea;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}

        .metric-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #e9ecef;
            transition: transform 0.2s;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}

        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .good {{
            color: #28a745;
        }}

        .warning {{
            color: #ffc107;
        }}

        .bad {{
            color: #dc3545;
        }}

        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}

        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}

        th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}

        tr:hover {{
            background-color: #f5f5f5;
        }}

        .feature-list {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}

        .feature-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
        }}

        .recommendations {{
            background: #e8f4f8;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin: 30px 0;
        }}

        .recommendations ul {{
            margin-left: 20px;
            margin-top: 15px;
        }}

        .analysis-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #e9ecef;
        }}

        .vif-high {{
            background-color: #ffebee;
            color: #c62828;
            font-weight: bold;
        }}

        .vif-ok {{
            background-color: #e8f5e9;
            color: #2e7d32;
        }}

        code {{
            background: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}

        .feature-code {{
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
            border: 1px solid #ddd;
            font-family: 'Courier New', monospace;
            color: #667eea;
            font-size: 14px;
            line-height: 1.5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Credit Card Fraud Detection Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </header>

        <h2>📊 Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Model Performance</div>
                <div class="metric-value good">{f1_score:.1%}</div>
                <div>F1 Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fraud Detection Rate</div>
                <div class="metric-value good">{fraud_caught:.1%}</div>
                <div>of all fraud cases</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">False Alarm Rate</div>
                <div class="metric-value">{false_positive_rate:.2%}</div>
                <div>of normal transactions</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Net Benefit</div>
                <div class="metric-value good">${net_benefit:,.0f}</div>
                <div>estimated savings</div>
            </div>
        </div>

        {f'''
        <div class="analysis-section" style="background: #fff3cd; border-left: 5px solid #ffc107;">
            <h3>🔑 Key Analysis Findings</h3>
            <ul>
                <li><strong>Multicollinearity:</strong> {len([r for r in analysis_results.get('vif_results', []) if r['VIF'] > 10])} features removed due to high VIF (>10)</li>
                <li><strong>Feature Selection:</strong> {len(analysis_results.get('selected_features', []))} features selected via RFE from {len(analysis_results.get('vif_results', []))} candidates</li>
                <li><strong>Top Predictors:</strong> {', '.join([r['feature'] for r in analysis_results.get('shap_importance', [])[:3]]) if analysis_results and 'shap_importance' in analysis_results else 'N/A'} show highest SHAP importance</li>
                <li><strong>Class Balance:</strong> SMOTE applied to address {data_stats['fraud_rate']:.2f}% fraud rate</li>
            </ul>

            <div style="margin-top: 15px; padding: 15px; background: white; border-radius: 5px;">
                <strong>Feature Engineering Pipeline:</strong>
                <div style="display: flex; align-items: center; justify-content: space-around; margin-top: 10px;">
                    <div style="text-align: center;">
                        <div style="font-size: 24px; color: #667eea;">30</div>
                        <div>Original Features</div>
                    </div>
                    <div>→</div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; color: #ffc107;">{len(analysis_results.get('vif_results', [])) - len([r for r in analysis_results.get('vif_results', []) if r.get('VIF', 0) > 10]) if analysis_results else 'N/A'}</div>
                        <div>After VIF Filter</div>
                    </div>
                    <div>→</div>
                    <div style="text-align: center;">
                        <div style="font-size: 24px; color: #28a745;">{len(analysis_results.get('selected_features', [])) if analysis_results else 'N/A'}</div>
                        <div>Final Selected</div>
                    </div>
                </div>
            </div>
        </div>
        ''' if analysis_results else ''}

        <h2>📈 Dataset Overview</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Transactions</div>
                <div class="metric-value">{data_stats['total_transactions']:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fraud Cases</div>
                <div class="metric-value bad">{data_stats['fraud_count']:,}</div>
                <div>{data_stats['fraud_rate']:.2f}% of total</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Normal Amount</div>
                <div class="metric-value">${data_stats['avg_normal_amount']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Fraud Amount</div>
                <div class="metric-value">${data_stats['avg_fraud_amount']:.2f}</div>
            </div>
        </div>

        <img src="data:image/png;base64,{eda_img}" alt="Exploratory Data Analysis">

        <h2>🤖 Model Performance</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Description</th>
            </tr>
            <tr>
                <td>AUC-ROC Score</td>
                <td class="good">{model_results['auc_score']:.3f}</td>
                <td>Overall model discrimination ability</td>
            </tr>
            <tr>
                <td>Average Precision</td>
                <td>{model_results['ap_score']:.3f}</td>
                <td>Precision-Recall AUC</td>
            </tr>
            <tr>
                <td>Precision (Fraud)</td>
                <td>{precision:.1%}</td>
                <td>When flagged as fraud, how often it's correct</td>
            </tr>
            <tr>
                <td>Recall (Fraud)</td>
                <td class="good">{recall:.1%}</td>
                <td>Percentage of actual fraud cases detected</td>
            </tr>
            <tr>
                <td>F1 Score (Fraud)</td>
                <td>{f1_score:.1%}</td>
                <td>Harmonic mean of precision and recall</td>
            </tr>
        </table>

        <img src="data:image/png;base64,{eval_img}" alt="Model Evaluation">

        {f'<img src="data:image/png;base64,{threshold_img}" alt="Threshold Analysis">' if threshold_img else ''}

        <h2>🔬 Feature Engineering & Analysis</h2>
        {f'''
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3>📊 VIF Analysis (Multicollinearity Check)</h3>
            <p>Variance Inflation Factor identifies multicollinear features:</p>

            {f'<img src="data:image/png;base64,{vif_img}" alt="VIF Analysis">' if vif_img else ''}

            <table style="margin-top: 20px;">
                <tr>
                    <th>Feature</th>
                    <th>VIF Score</th>
                    <th>Status</th>
                </tr>
                {''.join([f'''<tr>
                    <td>{r["Feature"]}</td>
                    <td>{r["VIF"]:.2f}</td>
                    <td class="{'bad' if r["VIF"] > 10 else 'good'}">
                        {'❌ High (Removed)' if r["VIF"] > 10 else '✅ Acceptable'}
                    </td>
                </tr>''' for r in (analysis_results['vif_results'][:10] if analysis_results else [])])}
            </table>
            <p style="margin-top: 10px;"><em>Features with VIF > 10 indicate multicollinearity and were removed.</em></p>
        </div>

        <div style="background: #e8f4f8; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3>🎯 Recursive Feature Elimination (RFE)</h3>
            <p>Selected <strong>{len(analysis_results['selected_features']) if analysis_results else 0}</strong> most important features using RFE:</p>
            <div class="feature-code">
                {', '.join(analysis_results['selected_features']) if analysis_results else 'No features selected'}
            </div>
        </div>

        <div style="background: #f0f8ff; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3>⚖️ Class Balancing with SMOTE</h3>
            <p>✅ <strong>SMOTE (Synthetic Minority Over-sampling Technique)</strong> was applied to balance the training data.</p>
            <p>This helps the model learn better patterns from the minority fraud class.</p>
        </div>
        ''' if analysis_results else '<p>No feature engineering details available.</p>'}

        {f'''
        <h2>🔍 SHAP Analysis (Model Explainability)</h2>
        <p>SHAP (SHapley Additive exPlanations) shows which features contribute most to fraud detection:</p>
        <img src="data:image/png;base64,{shap_img}" alt="SHAP Feature Importance">

        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3>Top SHAP Feature Importance Values</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>SHAP Importance</th>
                    <th>Impact</th>
                </tr>
                {''.join([f'''<tr>
                    <td>{i + 1}</td>
                    <td><strong>{r["feature"]}</strong></td>
                    <td>{r["importance"]:.4f}</td>
                    <td>{'🔴 High' if r["importance"] > 0.1 else '🟡 Medium' if r["importance"] > 0.05 else '🟢 Low'}</td>
                </tr>''' for i, r in enumerate(analysis_results.get('shap_importance', [])[:10]) if analysis_results])}
            </table>

            <div style="margin-top: 20px;">
                <p><strong>How to interpret:</strong></p>
                <ul>
                    <li>Higher importance values indicate features that strongly influence fraud predictions</li>
                    <li>SHAP values show the average impact of each feature across all predictions</li>
                    <li>Features with high importance should be monitored for data quality and drift</li>
                </ul>
            </div>
        </div>
        ''' if shap_img and analysis_results else ''}

        <h2>🔍 Confusion Matrix Analysis</h2>
        <table>
            <tr>
                <th>Outcome</th>
                <th>Count</th>
                <th>Business Impact</th>
            </tr>
            <tr>
                <td>True Positives (Fraud caught)</td>
                <td class="good">{tp:,}</td>
                <td class="good">+${tp * avg_fraud_amount:,.2f} saved</td>
            </tr>
            <tr>
                <td>True Negatives (Normal transactions)</td>
                <td>{tn:,}</td>
                <td>No impact</td>
            </tr>
            <tr>
                <td>False Positives (False alarms)</td>
                <td class="warning">{fp:,}</td>
                <td class="warning">-${false_alarm_cost:,.2f} in handling costs</td>
            </tr>
            <tr>
                <td>False Negatives (Missed fraud)</td>
                <td class="bad">{fn:,}</td>
                <td class="bad">-${fn * avg_fraud_amount:,.2f} potential loss</td>
            </tr>
        </table>

        <h2>📊 Top Features</h2>
        <div class="feature-list">
            <h3>Most Important Features for Fraud Detection:</h3>
            {''.join([f'<div class="feature-item"><span>{feat}</span><span>{imp:.4f}</span></div>'
                      for feat, imp in feature_importance[:10]])}
        </div>

        <div class="recommendations">
            <h2>💡 Recommendations</h2>
            <ul>
                <li><strong>Current Performance:</strong> The model successfully detects {fraud_caught:.1%} of fraudulent transactions while maintaining a low false positive rate of {false_positive_rate:.2%}.</li>
                <li><strong>Threshold Tuning:</strong> Consider adjusting the classification threshold based on business priorities. Lower thresholds catch more fraud but increase false alarms.</li>
                <li><strong>Feature Monitoring:</strong> Monitor the top features ({', '.join([f[0] for f in feature_importance[:3]])}) for data drift.</li>
                <li><strong>Retraining Schedule:</strong> Retrain the model monthly with new fraud patterns to maintain performance.</li>
                <li><strong>Cost-Benefit:</strong> Current configuration provides an estimated net benefit of ${net_benefit:,.2f} per test set.</li>
            </ul>
        </div>

        <div class="analysis-section">
            <h2>📋 Methodology Summary</h2>
            <p>This fraud detection model was built using industry best practices:</p>
            <ul>
                <li><strong>Multicollinearity Handling:</strong> VIF analysis to remove redundant features</li>
                <li><strong>Feature Selection:</strong> Recursive Feature Elimination (RFE) with Logistic Regression</li>
                <li><strong>Class Imbalance:</strong> SMOTE (Synthetic Minority Over-sampling Technique)</li>
                <li><strong>Model:</strong> Random Forest with balanced class weights</li>
                <li><strong>Explainability:</strong> SHAP (SHapley Additive exPlanations) for feature importance</li>
                <li><strong>Evaluation:</strong> Comprehensive metrics including precision-recall analysis</li>
            </ul>
        </div>

        <footer>
            <p>Credit Card Fraud Detection System | Model: Random Forest | Framework: scikit-learn</p>
        </footer>
    </div>
</body>
</html>
"""

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nHTML report generated: {output_path}")