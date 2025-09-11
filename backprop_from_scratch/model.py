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
    """ReLU activation function and its derivative"""
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

class Sigmoid(ActivationFunction):
    """Sigmoid activation function and its derivative"""
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))

    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        s = Sigmoid.forward(x)
        return s * (1 - s)

class Tanh(ActivationFunction):
    """Tanh activation function and its derivative"""
    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return np.tanh(x_clipped)
    
    @staticmethod
    def backward(x: np.ndarray) -> np.ndarray:
        x_clipped = np.clip(x, -500, 500)
        return 1 - np.tanh(x_clipped) ** 2

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
        # Initialize weights with Xavier/Glorot initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))

        # Store gradients for analysis
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # Calculate gradients
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
    """Mean Squared Error loss function"""
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.shape[0]

class BinaryCrossEntropy(LossFunction):
    """Binary Cross Entropy loss function"""
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Clip predictions to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped)) / y_true.shape[0]

class NeuralNetwork:
    """Neural network with manual backpropagation implementation"""
    def __init__(self):
        self.layers: List[Layer] = []
        self.loss_function = None
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'gradients': []
        }

    def add_layer(self, layer: Layer):
        """Add a layer to the network"""
        self.layers.append(layer)

    def set_loss(self, loss_function: LossFunction):
        """Set the loss function"""
        self.loss_function = loss_function

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation through all layers"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float):
        """Backward propagation through all layers"""
        # Calculate initial gradient from loss function
        gradient = self.loss_function.backward(y_true, y_pred)

        # Store gradient information for analysis
        gradient_norms = []

        # Propagate gradient backwards through layers
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

            # Record gradient norms for analysis
            if hasattr(layer, 'weight_gradients') and layer.weight_gradients is not None:
                gradient_norms.append(np.linalg.norm(layer.weight_gradients))

        # Store gradient information
        self.training_history['gradients'].append(gradient_norms)

    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """Perform one training step"""
        # Forward pass
        predictions = self.forward(X)

        # Calculate loss
        loss = self.loss_function.forward(y, predictions)

        # Backward pass
        self.backward(y, predictions, learning_rate)

        return loss

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float,
              batch_size: int = None, verbose: bool = True) -> dict:
        """Train the neural network"""
        if batch_size is None:
            batch_size = X.shape[0]

        n_samples = X.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            n_batches = 0

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                # Training step
                loss = self.train_step(X_batch, y_batch, learning_rate)
                epoch_loss += loss

                # Calculate accuracy for classification tasks
                predictions = self.forward(X_batch)
                if predictions.shape[1] == 1:  # Binary classification
                    predicted_classes = (predictions > 0.5).astype(int)
                    accuracy = np.mean(predicted_classes == y_batch)
                else:  # Multi-class classification
                    predicted_classes = np.argmax(predictions, axis=1)
                    true_classes = np.argmax(y_batch, axis=1)
                    accuracy = np.mean(predicted_classes == true_classes)

                epoch_accuracy += accuracy
                n_batches += 1

            # Average metrics for the epoch
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches

            # Store training history
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(avg_accuracy)

            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Accuracy: {avg_accuracy:.4f}")

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        return self.forward(X)

    def get_layer_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get weights and biases from all dense layers"""
        weights_and_biases = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                weights_and_biases.append((layer.weights.copy(), layer.bias.copy()))
        return weights_and_biases

# Example network architectures
def create_simple_classifier(input_size: int, hidden_size: int, output_size: int,
                           activation: str = 'relu') -> NeuralNetwork:
    """Create a simple 2-layer neural network for classification"""
    network = NeuralNetwork()

    # Choose activation function
    if activation.lower() == 'relu':
        act_func = ReLU()
    elif activation.lower() == 'sigmoid':
        act_func = Sigmoid()
    elif activation.lower() == 'tanh':
        act_func = Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    # Add layers
    network.add_layer(Dense(input_size, hidden_size))
    network.add_layer(Activation(act_func))
    network.add_layer(Dense(hidden_size, output_size))

    # Output activation depends on the task
    if output_size == 1:
        network.add_layer(Activation(Sigmoid()))  # Binary classification
        network.set_loss(BinaryCrossEntropy())
    else:
        network.add_layer(Activation(Sigmoid()))  # Multi-class (softmax would be better)
        network.set_loss(MeanSquaredError())

    return network

def create_regression_network(input_size: int, hidden_sizes: List[int], output_size: int) -> NeuralNetwork:
    """Create a neural network for regression tasks"""
    network = NeuralNetwork()

    # Input layer
    prev_size = input_size

    # Hidden layers
    for hidden_size in hidden_sizes:
        network.add_layer(Dense(prev_size, hidden_size))
        network.add_layer(Activation(ReLU()))
        prev_size = hidden_size

    # Output layer (no activation for regression)
    network.add_layer(Dense(prev_size, output_size))
    network.set_loss(MeanSquaredError())

    return network
