# generate_data.py - Generate realistic synthetic credit card data
# ===============================================================

import pandas as pd
import numpy as np


def generate_synthetic_data(n_samples=100000, fraud_rate=0.0017):
    """Generate synthetic credit card fraud data with realistic noise"""

    np.random.seed(42)
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud

    print(f"Generating {n_samples} transactions...")
    print(f"  Normal: {n_normal} ({(n_normal / n_samples) * 100:.2f}%)")
    print(f"  Fraud: {n_fraud} ({fraud_rate * 100:.2f}%)")

    # Initialize data dictionary
    data = {}

    # Generate V1-V28 features with overlapping distributions
    for i in range(1, 29):
        # Normal transactions
        normal_values = np.random.standard_normal(n_normal)

        # Fraud transactions - only slightly different from normal
        if i in [14, 17, 12, 10]:  # Only a few features are good indicators
            # Add small shift with lots of overlap
            fraud_shift = np.random.choice([-0.5, 0, 0.5], n_fraud, p=[0.4, 0.2, 0.4])
            fraud_values = np.random.normal(fraud_shift, 1.2, n_fraud)
        elif i in [3, 7]:
            # Even smaller difference
            fraud_values = np.random.normal(0.2, 1.1, n_fraud)
        else:
            # Most features have no discriminative power
            fraud_values = np.random.standard_normal(n_fraud)

        # Add noise to both classes
        noise_factor = 0.3
        normal_values += np.random.normal(0, noise_factor, n_normal)
        fraud_values += np.random.normal(0, noise_factor, n_fraud)

        # Combine
        data[f'V{i}'] = np.concatenate([normal_values, fraud_values])

    # Add some outliers to normal transactions (make it harder)
    outlier_indices = np.random.choice(n_normal, size=int(n_normal * 0.05), replace=False)
    for idx in outlier_indices:
        feature = np.random.choice([f'V{i}' for i in range(1, 29)])
        data[feature][idx] += np.random.normal(0, 3)  # Large deviation

    # Generate Time feature
    data['Time'] = np.sort(np.random.uniform(0, 172800, n_samples))

    # Generate Amount feature with significant overlap
    # Normal transactions
    normal_amounts = np.concatenate([
        np.random.lognormal(3.5, 2, int(n_normal * 0.7)),  # Most are medium amounts
        np.random.lognormal(2, 1.5, int(n_normal * 0.2)),  # Some small amounts
        np.random.lognormal(5, 1, int(n_normal * 0.1))  # Some large amounts
    ])
    normal_amounts = np.clip(normal_amounts, 0.01, 25000)

    # Fraud transactions - overlapping distribution
    fraud_amounts = np.concatenate([
        np.random.lognormal(2.5, 2, int(n_fraud * 0.5)),  # Half are small
        np.random.lognormal(3.5, 2, int(n_fraud * 0.3)),  # Some medium (like normal)
        np.random.lognormal(4, 1.5, int(n_fraud * 0.2))  # Some large
    ])
    fraud_amounts = np.clip(fraud_amounts, 0.01, 5000)

    # Mix amounts
    data['Amount'] = np.concatenate([normal_amounts[:n_normal], fraud_amounts[:n_fraud]])

    # Add Class label
    data['Class'] = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

    # Create DataFrame and shuffle
    df = pd.DataFrame(data)

    # Add some mislabeled data (noise in labels) - 1% label noise
    n_flip = int(n_samples * 0.01)
    flip_indices = np.random.choice(n_samples, n_flip, replace=False)
    df.loc[flip_indices, 'Class'] = 1 - df.loc[flip_indices, 'Class']

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Round amounts
    df['Amount'] = df['Amount'].round(2)

    # Save to CSV
    df.to_csv('creditcard.csv', index=False)

    print(f"\n✓ Data saved to creditcard.csv")
    print(f"  Shape: {df.shape}")
    print(f"  Avg amount: ${df['Amount'].mean():.2f}")
    print(f"  Fraud amount: ${df[df['Class'] == 1]['Amount'].mean():.2f}")
    print(f"  Label noise added: {n_flip} samples")
    print("\nNote: This data is intentionally noisy for realistic testing")


if __name__ == "__main__":
    # Generate with more samples to ensure fraud in test set
    generate_synthetic_data(n_samples=100000, fraud_rate=0.005)  # 0.5% fraud rate