import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class ActivationFunction:
    """Base class for activation functions"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(ActivationFunction):
    """ReLU: f(x) = max(0, x)"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid(ActivationFunction):
    """Sigmoid: f(x) = 1 / (1 + exp(-x))"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        s = Sigmoid.forward(x)
        return s * (1 - s)


class Tanh(ActivationFunction):
    """Tanh: f(x) = tanh(x)"""
    
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -500, 500))
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(np.clip(x, -500, 500)) ** 2

class Layer:
    """Base class for neural network layers"""
    
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        raise NotImplementedError


class Dense(Layer):
    """Fully connected layer with manual backpropagation"""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        # Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        
        # For gradient tracking
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Compute gradients
        self.weight_gradients = np.dot(self.input.T, output_gradient)
        self.bias_gradients = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Update parameters
        self.weights -= learning_rate * self.weight_gradients
        self.bias -= learning_rate * self.bias_gradients

        return input_gradient

class Activation(Layer):
    """Activation layer wrapper"""
    
    def __init__(self, activation_function: ActivationFunction):
        super().__init__()
        self.activation = activation_function

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = self.activation.forward(input_data)
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        return output_gradient * self.activation.backward(self.input)


class LossFunction:
    """Base class for loss functions"""
    
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """MSE Loss: L = mean((y_true - y_pred)^2)"""
    
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.shape[0]


class BinaryCrossEntropy(LossFunction):
    """Binary Cross Entropy Loss"""
    
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]

class NeuralNetwork:
    """Neural network with manual backpropagation"""
    
    def __init__(self):
        self.layers: List[Layer] = []
        self.loss_function = None
        self.history = {'loss': [], 'accuracy': [], 'gradients': []}

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def set_loss(self, loss_function: LossFunction):
        self.loss_function = loss_function

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float):
        """Backward pass through all layers"""
        gradient = self.loss_function.backward(y_true, y_pred)
        gradient_norms = []

        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
            if hasattr(layer, 'weight_gradients') and layer.weight_gradients is not None:
                gradient_norms.append(np.linalg.norm(layer.weight_gradients))

        self.history['gradients'].append(gradient_norms)

    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """Single training step"""
        predictions = self.forward(X)
        loss = self.loss_function.forward(y, predictions)
        self.backward(y, predictions, learning_rate)
        return loss

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float,
              batch_size: int = None, verbose: bool = True) -> dict:
        """Train the network"""
        if batch_size is None:
            batch_size = X.shape[0]

        n_samples = X.shape[0]

        for epoch in range(epochs):
            epoch_loss = epoch_accuracy = n_batches = 0

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch, y_batch = X_shuffled[i:end_idx], y_shuffled[i:end_idx]

                # Train step
                loss = self.train_step(X_batch, y_batch, learning_rate)
                epoch_loss += loss

                # Calculate accuracy
                predictions = self.forward(X_batch)
                if predictions.shape[1] == 1:  # Binary
                    predicted_classes = (predictions > 0.5).astype(int)
                    accuracy = np.mean(predicted_classes == y_batch)
                else:  # Multi-class
                    predicted_classes = np.argmax(predictions, axis=1)
                    true_classes = np.argmax(y_batch, axis=1)
                    accuracy = np.mean(predicted_classes == true_classes)

                epoch_accuracy += accuracy
                n_batches += 1

            # Store metrics
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(avg_accuracy)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.6f} | Acc: {avg_accuracy:.4f}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def get_layer_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get weights and biases from dense layers"""
        return [(layer.weights.copy(), layer.bias.copy()) 
                for layer in self.layers if isinstance(layer, Dense)]

def create_classifier(input_size: int, hidden_size: int, output_size: int,
                     activation: str = 'relu') -> NeuralNetwork:
    """Create a simple classifier network"""
    network = NeuralNetwork()

    # Choose activation
    activation_map = {'relu': ReLU(), 'sigmoid': Sigmoid(), 'tanh': Tanh()}
    if activation.lower() not in activation_map:
        raise ValueError(f"Unsupported activation: {activation}")
    act_func = activation_map[activation.lower()]

    # Build network
    network.add_layer(Dense(input_size, hidden_size))
    network.add_layer(Activation(act_func))
    network.add_layer(Dense(hidden_size, output_size))
    
    # Output layer and loss
    if output_size == 1:
        network.add_layer(Activation(Sigmoid()))
        network.set_loss(BinaryCrossEntropy())
    else:
        network.add_layer(Activation(Sigmoid()))
        network.set_loss(MeanSquaredError())

    return network


def create_regression_network(input_size: int, hidden_sizes: List[int], 
                             output_size: int = 1) -> NeuralNetwork:
    """Create a regression network"""
    network = NeuralNetwork()
    prev_size = input_size

    # Hidden layers
    for hidden_size in hidden_sizes:
        network.add_layer(Dense(prev_size, hidden_size))
        network.add_layer(Activation(ReLU()))
        prev_size = hidden_size

    # Output layer
    network.add_layer(Dense(prev_size, output_size))
    network.set_loss(MeanSquaredError())

    return network
