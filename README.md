# Credit Card Fraud Detection - ML Pipeline

A comprehensive machine learning pipeline for credit card fraud detection with advanced feature engineering and explainable AI.

## Features

- **Multicollinearity Analysis**: VIF (Variance Inflation Factor) to remove redundant features
- **Feature Selection**: Recursive Feature Elimination (RFE) 
- **Class Balancing**: SMOTE for handling imbalanced data
- **Model Training**: Random Forest with optimized parameters
- **Explainability**: SHAP analysis for feature importance
- **Comprehensive Reporting**: HTML report with all visualizations and metrics

## Project Structure

```
fraud-detection/
├── train_model.py      # Main training pipeline
├── predict.py          # Make predictions
├── config.py           # Configuration settings
├── data_utils.py       # Data processing, VIF, RFE, SMOTE
├── model_utils.py      # Model training, evaluation, SHAP
├── report_generator.py # HTML report generation
├── generate_data.py    # Synthetic data generator
├── run_pipeline.py     # Interactive runner
└── requirements.txt    # Dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- statsmodels (for VIF)
- imbalanced-learn (for SMOTE)
- shap (for explainability)
- joblib

### 2. Get Data

**Option A:** Download from Kaggle
- [Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Option B:** Generate synthetic data
```bash
python generate_data.py
```

### 3. Train Model
```bash
python train_model.py
```

This will:
1. Load and analyze data
2. Calculate VIF and remove multicollinear features
3. Perform RFE feature selection (top 15 features)
4. Apply SMOTE to balance classes
5. Train Random Forest model
6. Generate SHAP explanations
7. Create comprehensive HTML report

### 4. View Results

Open `fraud_report.html` in your browser to see:
- Executive summary with key metrics
- Data analysis visualizations
- VIF analysis results and chart
- Selected features from RFE
- Model performance metrics
- SHAP feature importance
- Business impact analysis
- Recommendations

## Pipeline Workflow

```
Data Loading → EDA → VIF Analysis → Feature Selection (RFE) → 
SMOTE Balancing → Model Training → SHAP Analysis → HTML Report
```

## Output Files

- `fraud_model.pkl` - Trained model with scaler and metadata
- `fraud_report.html` - Complete analysis report
- `figures/` - All visualizations:
  - `eda_analysis.png` - Exploratory data analysis
  - `vif_analysis.png` - VIF scores chart
  - `model_evaluation.png` - Performance metrics
  - `shap_importance.png` - SHAP feature importance
  - `threshold_analysis.png` - Precision/Recall vs threshold

## Making Predictions

Single prediction:
```python
python predict.py
```

Batch predictions:
```bash
python predict.py transactions.csv
```

## Configuration

Edit `config.py` to customize:
- Model parameters
- Number of features to select
- Train/test split ratio
- Output paths

## Key Improvements

1. **Feature Engineering**:
   - Removes features with VIF > 10
   - Selects best 15 features using RFE
   - Handles class imbalance with SMOTE

2. **Model Explainability**:
   - SHAP analysis for global feature importance
   - Identifies top predictive features
   - Helps understand model decisions

3. **Comprehensive Reporting**:
   - All analyses in single HTML file
   - Professional visualizations
   - Business metrics and recommendations

## Troubleshooting

If you get import errors:
```bash
pip install --upgrade -r requirements.txt
```

For better model performance with synthetic data:
```python
# In generate_data.py, increase samples and fraud rate
generate_synthetic_data(n_samples=100000, fraud_rate=0.02)
```

## Performance

Typical results:
- **Precision**: ~70-90% (varies with data quality)
- **Recall**: ~70-95%
- **F1-Score**: ~70-90%
- **AUC-ROC**: ~0.85-0.99

Note: Performance depends heavily on data quality. Synthetic data may show lower scores due to added noise for realism.