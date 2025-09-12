import numpy as np

class LinearRegressionModel:
    def __init__(self):
        """
        Initialize linear regression model: y = W * x + B
        Parameters are randomly initialized
        """
        # Initialize weight and bias with small random values
        self.W = np.random.normal(0, 0.1, size=(1,))
        self.B = np.random.normal(0, 0.1, size=(1,))

        # Store gradients for optimization
        self.dW = None
        self.dB = None

    def forward(self, X):
        """
        Forward pass: compute predictions
        Args:
            X: input features (N, 1)
        Returns:
            predictions: y_pred = W * X + B (N, 1)
        """
        return np.dot(X, self.W).reshape(-1, 1) + self.B.reshape(1, 1)

    def compute_loss(self, y_pred, y_true):
        """
        Compute Mean Squared Error loss
        Args:
            y_pred: predicted values (N, 1)
            y_true: true values (N, 1)
        Returns:
            loss: MSE loss (scalar)
        """
        N = y_true.shape[0]
        loss = np.sum((y_pred - y_true) ** 2) / N
        return loss

    def backward(self, X, y_pred, y_true):
        """
        Compute gradients using chain rule
        For MSE loss: L = (1/N) * sum((y_pred - y_true)^2)
        dL/dW = (2/N) * sum((y_pred - y_true) * X)
        dL/dB = (2/N) * sum(y_pred - y_true)
        """
        N = y_true.shape[0]
        error = y_pred - y_true

        # Compute gradients
        self.dW = (2.0 / N) * np.dot(X.T, error)
        self.dB = (2.0 / N) * np.sum(error, axis=0, keepdims=True)

    def update_parameters(self, learning_rate):
        """
        Update parameters using gradient descent
        W = W - learning_rate * dW
        B = B - learning_rate * dB
        """
        self.W -= learning_rate * self.dW.flatten()
        self.B -= learning_rate * self.dB.flatten()

    def get_parameters(self):
        """Return current weight and bias values"""
        return self.W[0], self.B[0]
