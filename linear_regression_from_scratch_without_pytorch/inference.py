import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from model import LinearRegressionModel
from generate_data import load_test_data, check_data_files

def evaluate_model(data_dir='data', model_path=None):
    """Evaluate trained linear regression model on test data"""

    # Check files
    exists, message = check_data_files(data_dir)
    if not exists:
        print(f"Error: {message}")
        return None

    if model_path is None:
        model_path = os.path.join(data_dir, 'linear_model.pkl')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    # Load data and model
    X_test, y_test, y_true_test, W_true, B_true = load_test_data(data_dir)

    # Load model parameters
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Create model and set parameters
    model = LinearRegressionModel()
    model.W = model_data['W']
    model.B = model_data['B']

    # Make predictions
    y_pred = model.forward(X_test)

    # Get learned parameters
    learned_W, learned_B = model.get_parameters()

    # Compute metrics
    mse = compute_mse(y_pred, y_test)
    mae = compute_mae(y_pred, y_test)
    r2 = compute_r2(y_pred, y_test)

    print(f"Test Results | Samples: {len(X_test)} | MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    print(f"Parameters   | True: W={W_true:.3f}, B={B_true:.3f} | Learned: W={learned_W:.3f}, B={learned_B:.3f}")

    # Visualization
    plt.figure(figsize=(10, 4))
    X_flat = X_test.flatten()
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true_test.flatten()

    plt.scatter(X_flat, y_test_flat, alpha=0.5, label='Test Data', s=20)
    plt.scatter(X_flat, y_pred_flat, alpha=0.7, label='Predictions', s=20)
    plt.plot(X_flat, y_true_flat, 'g-', label='True Function', alpha=0.8)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Results (No PyTorch)')
    plt.legend()
    plt.grid(True)

    results_path = os.path.join(data_dir, 'test_results.png')
    plt.savefig(results_path)
    plt.show()
    print(f"Results saved to {results_path}")

    return {
        'mse': mse, 'mae': mae, 'r2': r2,
        'learned_W': learned_W, 'learned_B': learned_B,
        'true_W': W_true, 'true_B': B_true
    }

def compute_mse(y_pred, y_true):
    """Compute Mean Squared Error"""
    return np.mean((y_pred - y_true) ** 2)

def compute_mae(y_pred, y_true):
    """Compute Mean Absolute Error"""
    return np.mean(np.abs(y_pred - y_true))

def compute_r2(y_pred, y_true):
    """Compute R-squared coefficient of determination"""
    ss_res = np.sum((y_true - y_pred) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)

def compare_predictions(data_dir='data', model_path=None):
    """
    Compare predictions with true function across the input range
    This gives a better sense of how well the model generalizes
    """
    if model_path is None:
        model_path = os.path.join(data_dir, 'linear_model.pkl')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = LinearRegressionModel()
    model.W = model_data['W']
    model.B = model_data['B']

    # Load metadata to get true parameters
    _, _, _, W_true, B_true = load_test_data(data_dir)

    # Generate a range of X values for comparison
    X_range = np.linspace(-20, 20, 100).reshape(-1, 1)

    # Compute predictions and true values
    y_pred_range = model.forward(X_range)
    y_true_range = W_true * X_range + B_true

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(X_range.flatten(), y_true_range.flatten(), 'g-', linewidth=2, label=f'True: y = {W_true:.2f}x + {B_true:.2f}')
    plt.plot(X_range.flatten(), y_pred_range.flatten(), 'r--', linewidth=2, label=f'Learned: y = {model.W[0]:.2f}x + {model.B[0]:.2f}')

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('True vs Learned Function')
    plt.legend()
    plt.grid(True)

    comparison_path = os.path.join(data_dir, 'function_comparison.png')
    plt.savefig(comparison_path)
    plt.show()
    print(f"Function comparison saved to {comparison_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained linear regression model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model (default: data_dir/linear_model.pkl)')
    parser.add_argument('--compare', action='store_true', help='Show function comparison plot')

    args = parser.parse_args()

    evaluate_model(args.data_dir, args.model_path)

    if args.compare:
        print("\nGenerating function comparison...")
        compare_predictions(args.data_dir, args.model_path)
