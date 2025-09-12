import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from model import LinearRegressionModel
from generate_data import load_train_data, check_data_files

def train_model(data_dir='data', learning_rate=0.01, epochs=100):
    """Train linear regression model without PyTorch"""

    # Check data files
    exists, message = check_data_files(data_dir)
    if not exists:
        print(f"Error: {message}")
        print("Run: python generate_data.py to create data files")
        return None

    # Load data
    X, y, y_true, W_true, B_true = load_train_data(data_dir)

    # Initialize model
    model = LinearRegressionModel()

    print(f"Samples: {len(X)} | True: W={W_true:.3f}, B={B_true:.3f}")
    print(f"Initial: W={model.W[0]:.3f}, B={model.B[0]:.3f}")

    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.forward(X)

        # Compute loss
        loss = model.compute_loss(y_pred, y)
        losses.append(loss)

        # Backward pass (compute gradients)
        model.backward(X, y_pred, y)

        # Update parameters
        model.update_parameters(learning_rate)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1:3d}/{epochs}, Loss: {loss:.4f}')

    # Final results
    final_W, final_B = model.get_parameters()
    print(f"\nResults: Learned W={final_W:.3f}, B={final_B:.3f} | True W={W_true:.3f}, B={B_true:.3f}")

    # Save model using pickle (since we don't have PyTorch's state_dict)
    model_path = os.path.join(data_dir, 'linear_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'W': model.W,
            'B': model.B,
            'final_loss': losses[-1]
        }, f)

    # Plot and save loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_curve_path = os.path.join(data_dir, 'loss_curve.png')
    plt.savefig(loss_curve_path)
    plt.show()
    print(f"Model & loss curve saved to {data_dir}/")

    return model, losses

def manual_gradient_check():
    """
    Verify our gradient computation is correct by comparing with numerical gradients
    This is a useful debugging tool for gradient-based optimization
    """
    print("Running gradient check...")

    # Create simple test data
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[3.0], [5.0], [7.0]])  # y = 2x + 1

    model = LinearRegressionModel()
    model.W = np.array([1.5])  # Set to specific value for testing
    model.B = np.array([0.5])

    # Analytical gradients
    y_pred = model.forward(X)
    model.backward(X, y_pred, y)
    analytical_dW = model.dW[0, 0]  # Extract scalar from 2D array
    analytical_dB = model.dB[0, 0]  # Extract scalar from 2D array

    # Numerical gradients (finite differences)
    epsilon = 1e-7

    # dW numerical
    model.W[0] += epsilon
    y_pred_plus = model.forward(X)
    loss_plus = model.compute_loss(y_pred_plus, y)

    model.W[0] -= 2 * epsilon
    y_pred_minus = model.forward(X)
    loss_minus = model.compute_loss(y_pred_minus, y)

    numerical_dW = (loss_plus - loss_minus) / (2 * epsilon)
    model.W[0] += epsilon  # Reset W

    # dB numerical
    model.B[0] += epsilon
    y_pred_plus = model.forward(X)
    loss_plus = model.compute_loss(y_pred_plus, y)

    model.B[0] -= 2 * epsilon
    y_pred_minus = model.forward(X)
    loss_minus = model.compute_loss(y_pred_minus, y)

    numerical_dB = (loss_plus - loss_minus) / (2 * epsilon)
    model.B[0] += epsilon  # Reset B

    print(f"dW - Analytical: {analytical_dW:.6f}, Numerical: {numerical_dW:.6f}, Diff: {abs(analytical_dW - numerical_dW):.2e}")
    print(f"dB - Analytical: {analytical_dB:.6f}, Numerical: {numerical_dB:.6f}, Diff: {abs(analytical_dB - numerical_dB):.2e}")

    if abs(analytical_dW - numerical_dW) < 1e-5 and abs(analytical_dB - numerical_dB) < 1e-5:
        print("✓ Gradient check passed!")
    else:
        print("✗ Gradient check failed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train linear regression model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--grad_check', action='store_true', help='Run gradient check before training')

    args = parser.parse_args()

    if args.grad_check:
        manual_gradient_check()
        print()

    train_model(args.data_dir, args.learning_rate, args.epochs)
