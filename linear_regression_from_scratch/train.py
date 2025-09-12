import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import argparse
from model import LinearRegressionModel
from generate_data import load_train_data, check_data_files

def train_model(data_dir='data', learning_rate=0.01, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check data files
    exists, message = check_data_files(data_dir)
    if not exists:
        print(f"Error: {message}")
        print("Run: python generate_data.py to create data files")
        return None

    # Load data
    X, y, y_true, W_true, B_true = load_train_data(data_dir)
    X, y = X.to(device), y.to(device)

    # Initialize model
    model = LinearRegressionModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print(f"Device: {device} | Samples: {len(X)} | True: W={W_true:.3f}, B={B_true:.3f}")

    # Training loop
    losses = []
    for epoch in range(epochs):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1:3d}/{epochs}, Loss: {loss.item():.4f}')

    # Results
    final_W, final_B = model.linear.weight.item(), model.linear.bias.item()
    print(f"\nResults: Learned W={final_W:.3f}, B={final_B:.3f} | True W={W_true:.3f}, B={B_true:.3f}")

    # Save model and plot
    model_path = os.path.join(data_dir, 'linear_model.pth')
    torch.save(model.state_dict(), model_path)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train linear regression model')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

    args = parser.parse_args()

    train_model(args.data_dir, args.learning_rate, args.epochs)