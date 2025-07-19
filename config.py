# config.py - Configuration settings
# ==================================

# Data settings
DATA_PATH = 'creditcard.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model settings
MODEL_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced'
}

# Output paths
MODEL_PATH = 'fraud_model.pkl'
REPORT_PATH = 'fraud_report.html'
FIGURES_DIR = 'figures/'

# Feature selection
TOP_FEATURES = 15

# Imbalance thresholds (built into ImbalanceDetector)
# Critical: <1% minority, <50 samples, >100:1 ratio
# High: <5% minority, <100 samples, >20:1 ratio  
# Medium: <10% minority, <200 samples, >10:1 ratio

# Fraud detection thresholds
FRAUD_THRESHOLDS = {
    'min_fraud_samples': 100,
    'max_fpr': 0.01,
    'min_precision': 0.8,
    'min_recall': 0.7
}