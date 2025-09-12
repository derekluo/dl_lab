import numpy as np
import pandas as pd
import os
import argparse

def check_data_files(data_dir='data'):
    """Check if required data files exist"""
    files = ['train_data.csv', 'test_data.csv', 'metadata.csv']
    for file in files:
        if not os.path.exists(os.path.join(data_dir, file)):
            return False, f"Missing {file} in {data_dir}/"
    return True, "All data files found"

def generate_data(num_samples=100, train_split=0.8, data_dir='data'):
    """
    Generates synthetic data for a known linear relationship: y = 2x + 1
    Saves training and testing data to files
    Uses NumPy instead of PyTorch tensors
    """
    W_true = 2.0
    B_true = 1.0

    # Generate random input features
    np.random.seed(42)  # For reproducibility
    X = np.random.normal(0, 10, size=(num_samples, 1))

    # Generate true output (without noise)
    y_true = W_true * X + B_true

    # Add noise to create target values
    noise = np.random.normal(0, 2, size=(num_samples, 1))
    y = y_true + noise

    # Split data into train and test sets
    num_train = int(num_samples * train_split)

    X_train = X[:num_train]
    y_train = y[:num_train]
    y_true_train = y_true[:num_train]

    X_test = X[num_train:]
    y_test = y[num_train:]
    y_true_test = y_true[num_train:]

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Convert to flattened arrays for CSV saving
    X_train_flat = X_train.flatten()
    y_train_flat = y_train.flatten()
    y_true_train_flat = y_true_train.flatten()

    X_test_flat = X_test.flatten()
    y_test_flat = y_test.flatten()
    y_true_test_flat = y_true_test.flatten()

    # Save training data to CSV
    train_df = pd.DataFrame({
        'X': X_train_flat,
        'y': y_train_flat,
        'y_true': y_true_train_flat
    })
    train_csv_path = os.path.join(data_dir, 'train_data.csv')
    train_df.to_csv(train_csv_path, index=False)

    # Save testing data to CSV
    test_df = pd.DataFrame({
        'X': X_test_flat,
        'y': y_test_flat,
        'y_true': y_true_test_flat
    })
    test_csv_path = os.path.join(data_dir, 'test_data.csv')
    test_df.to_csv(test_csv_path, index=False)

    # Save metadata (true parameters) to a separate file
    metadata = {
        'W_true': W_true,
        'B_true': B_true,
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test)
    }
    metadata_df = pd.DataFrame([metadata])
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)

    print(f"Generated {num_samples} samples")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Data saved to {data_dir}/ directory")

    return X_train, y_train, y_true_train, X_test, y_test, y_true_test

def load_data(data_dir='data', split='train'):
    """Load data from CSV file - can load either train or test split"""
    data_path = os.path.join(data_dir, f'{split}_data.csv')
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    # Load data
    df = pd.read_csv(data_path)
    X = df['X'].values.reshape(-1, 1)  # Convert to column vector
    y = df['y'].values.reshape(-1, 1)  # Convert to column vector
    y_true = df['y_true'].values.reshape(-1, 1)  # Convert to column vector

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    W_true = metadata_df['W_true'].iloc[0]
    B_true = metadata_df['B_true'].iloc[0]

    return X, y, y_true, W_true, B_true

def load_train_data(data_dir='data'):
    """Load training data - kept for backward compatibility"""
    return load_data(data_dir, 'train')

def load_test_data(data_dir='data'):
    """Load test data - kept for backward compatibility"""
    return load_data(data_dir, 'test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate linear regression dataset')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--train_split', type=float, default=0.8, help='Training data split ratio')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to save data files')

    args = parser.parse_args()

    generate_data(args.num_samples, args.train_split, args.data_dir)
