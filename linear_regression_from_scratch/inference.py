import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from model import LinearRegressionModel
from generate_data import load_test_data, check_data_files

def evaluate_model(data_dir='data', model_path=None):
    """Evaluate trained linear regression model on test data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check files
    exists, message = check_data_files(data_dir)
    if not exists:
        print(f"Error: {message}")
        return None

    if model_path is None:
        model_path = os.path.join(data_dir, 'linear_model.pth')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    # Load data and model
    X_test, y_test, y_true_test, W_true, B_true = load_test_data(data_dir)
    X_test, y_test = X_test.to(device), y_test.to(device)

    model = LinearRegressionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predictions and metrics
    with torch.no_grad():
        y_pred = model(X_test)

    learned_W, learned_B = model.linear.weight.item(), model.linear.bias.item()
    mse = nn.MSELoss()(y_pred, y_test).item()
    mae = nn.L1Loss()(y_pred, y_test).item()

    y_test_np, y_pred_np = y_test.cpu().numpy(), y_pred.cpu().numpy()
    r2 = 1 - np.sum((y_test_np - y_pred_np)**2) / np.sum((y_test_np - y_test_np.mean())**2)

    print(f"Test Results | Samples: {len(X_test)} | MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    print(f"Parameters   | True: W={W_true:.3f}, B={B_true:.3f} | Learned: W={learned_W:.3f}, B={learned_B:.3f}")

    # Simple visualization
    plt.figure(figsize=(10, 4))
    X_np = X_test.cpu().numpy().flatten()
    y_test_flat, y_pred_flat = y_test_np.flatten(), y_pred_np.flatten()
    y_true_flat = y_true_test.numpy().flatten()

    plt.scatter(X_np, y_test_flat, alpha=0.5, label='Test Data', s=20)
    plt.scatter(X_np, y_pred_flat, alpha=0.7, label='Predictions', s=20)
    plt.plot(X_np, y_true_flat, 'g-', label='True Function', alpha=0.8)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Results')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained linear regression model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model (default: data_dir/linear_model.pth)')

    args = parser.parse_args()
    evaluate_model(args.data_dir, args.model_path)
