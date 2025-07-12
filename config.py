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