# run_pipeline.py - Run the complete fraud detection pipeline
# ===========================================================

import os
import sys
import subprocess


def check_requirements():
    """Check if required packages are installed"""
    try:
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        import joblib
        return True
    except ImportError:
        return False


def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def check_data():
    """Check if data file exists"""
    if os.path.exists('creditcard.csv'):
        return True
    return False


def generate_data():
    """Generate synthetic data"""
    print("\nGenerating synthetic data...")
    subprocess.check_call([sys.executable, "generate_data.py"])


def train_model():
    """Run the training pipeline"""
    print("\nRunning training pipeline...")
    subprocess.check_call([sys.executable, "train_model.py"])


def main():
    """Run complete pipeline"""
    print("=" * 60)
    print("Credit Card Fraud Detection - Pipeline Runner")
    print("=" * 60)

    # 1. Check requirements
    print("\n1. Checking requirements...")
    if not check_requirements():
        print("   Missing packages detected.")
        response = input("   Install requirements? (y/n): ")
        if response.lower() == 'y':
            install_requirements()
        else:
            print("   Please install requirements manually: pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("   ✓ All requirements satisfied")

    # 2. Check data
    print("\n2. Checking for data...")
    if not check_data():
        print("   No data found.")
        response = input("   Generate synthetic data? (y/n): ")
        if response.lower() == 'y':
            generate_data()
        else:
            print("   Please download creditcard.csv from Kaggle")
            print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            sys.exit(1)
    else:
        print("   ✓ Data found")

    # 3. Train model
    print("\n3. Training model...")
    response = input("   Proceed with training? (y/n): ")
    if response.lower() == 'y':
        train_model()

        print("\n" + "=" * 60)
        print("✅ Pipeline completed successfully!")
        print("\nResults:")
        print("  - Model: fraud_model.pkl")
        print("  - Report: fraud_report.html")
        print("  - Figures: figures/")
        print("\nOpen fraud_report.html in your browser to view results")
        print("=" * 60)
    else:
        print("   Training cancelled")


if __name__ == "__main__":
    main()