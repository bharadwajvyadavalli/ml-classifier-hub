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
    
    # Check for data imbalance
    total_samples = data_stats['total_transactions']
    fraud_count = data_stats['fraud_count']
    minority_ratio = fraud_count / total_samples
    imbalance_ratio = (total_samples - fraud_count) / fraud_count if fraud_count > 0 else float('inf')
    
    # Define imbalance red flags
    imbalance_flags = []
    if minority_ratio <= 0.01:
        imbalance_flags.append("CRITICAL: Extreme imbalance (< 1% fraud)")
    elif minority_ratio <= 0.05:
        imbalance_flags.append("HIGH: Severe imbalance (< 5% fraud)")
    elif minority_ratio <= 0.1:
        imbalance_flags.append("MEDIUM: Moderate imbalance (< 10% fraud)")
        
    if fraud_count < 50:
        imbalance_flags.append("CRITICAL: Too few fraud samples (< 50)")
    elif fraud_count < 100:
        imbalance_flags.append("HIGH: Low fraud samples (< 100)")
        
    if imbalance_ratio > 100:
        imbalance_flags.append("CRITICAL: Very high imbalance ratio (> 100:1)")
    elif imbalance_ratio > 20:
        imbalance_flags.append("HIGH: High imbalance ratio (> 20:1)")

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
    eval_img = encode_image(f'{figures_dir}model_evaluation.png')
    shap_img = encode_image(f'{figures_dir}shap_importance.png') if os.path.exists(
        f'{figures_dir}shap_importance.png') else None

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

        .analysis-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #e9ecef;
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

        <h2>📊 Top Features</h2>
        <div class="feature-list">
            <h3>Most Important Features for Fraud Detection:</h3>
            {''.join([f'<div class="feature-item"><span>{feat}</span><span>{imp:.4f}</span></div>'
                      for feat, imp in feature_importance[:10]])}
        </div>

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
        </div>
        ''' if shap_img and analysis_results else ''}

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