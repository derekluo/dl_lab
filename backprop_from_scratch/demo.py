import numpy as np
import matplotlib.pyplot as plt
from model import Dense, Activation, ReLU, Sigmoid, MeanSquaredError, NeuralNetwork

class DetailedNeuralNetwork(NeuralNetwork):
    """Extended neural network that provides detailed backpropagation step-by-step analysis"""

    def __init__(self):
        super().__init__()
        self.step_by_step_info = []

    def detailed_forward(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """Forward pass with detailed logging"""
        if verbose:
            print("=" * 60)
            print("FORWARD PROPAGATION")
            print("=" * 60)

        output = X
        layer_outputs = [X]

        if verbose:
            print(f"Input shape: {X.shape}")
            print(f"Input values:\n{X}\n")

        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
            layer_outputs.append(output.copy())

            if verbose:
                layer_type = type(layer).__name__
                if isinstance(layer, Dense):
                    print(f"Layer {i+1} ({layer_type}):")
                    print(f"  Weights shape: {layer.weights.shape}")
                    print(f"  Weights:\n{layer.weights}")
                    print(f"  Bias: {layer.bias}")
                    print(f"  Output shape: {output.shape}")
                    print(f"  Output values:\n{output}\n")
                elif isinstance(layer, Activation):
                    activation_name = type(layer.activation).__name__
                    print(f"Layer {i+1} ({layer_type} - {activation_name}):")
                    print(f"  Output shape: {output.shape}")
                    print(f"  Output values:\n{output}\n")

        self.layer_outputs = layer_outputs
        return output

    def detailed_backward(self, y_true: np.ndarray, y_pred: np.ndarray, learning_rate: float, verbose: bool = True):
        """Backward pass with detailed logging"""
        if verbose:
            print("=" * 60)
            print("BACKWARD PROPAGATION")
            print("=" * 60)

        # Calculate initial gradient from loss function
        gradient = self.loss_function.backward(y_true, y_pred)

        if verbose:
            loss_value = self.loss_function.forward(y_true, y_pred)
            print(f"Loss value: {loss_value:.6f}")
            print(f"Initial gradient (dL/dy_pred):\n{gradient}\n")

        # Store gradient information
        gradient_info = []

        # Propagate gradient backwards through layers
        for i, layer in enumerate(reversed(self.layers)):
            layer_index = len(self.layers) - 1 - i

            if verbose:
                layer_type = type(layer).__name__
                print(f"Layer {layer_index + 1} ({layer_type}) - Backward pass:")
                print(f"  Input gradient shape: {gradient.shape}")
                print(f"  Input gradient:\n{gradient}")

            # Store pre-backward state
            if isinstance(layer, Dense):
                old_weights = layer.weights.copy()
                old_bias = layer.bias.copy()

            # Backward pass
            gradient = layer.backward(gradient, learning_rate)

            if verbose:
                if isinstance(layer, Dense):
                    print(f"  Weight gradients:\n{layer.weight_gradients}")
                    print(f"  Bias gradients: {layer.bias_gradients}")
                    print(f"  Weight update: -{learning_rate} * gradients")
                    print(f"  Old weights:\n{old_weights}")
                    print(f"  New weights:\n{layer.weights}")
                    print(f"  Weight change:\n{layer.weights - old_weights}")

                    # Store gradient information
                    gradient_info.append({
                        'layer': layer_index,
                        'weight_gradients': layer.weight_gradients.copy(),
                        'bias_gradients': layer.bias_gradients.copy(),
                        'weight_change': layer.weights - old_weights,
                        'bias_change': layer.bias - old_bias
                    })

                print(f"  Output gradient shape: {gradient.shape}")
                print(f"  Output gradient:\n{gradient}\n")

        self.step_by_step_info.append(gradient_info)

def simple_xor_demo():
    """Demonstrate backpropagation on XOR problem - classic non-linearly separable problem"""
    print("=" * 70)
    print("XOR PROBLEM DEMONSTRATION")
    print("=" * 70)
    print("XOR is a classic example that requires a hidden layer to solve")
    print("Input patterns: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0")
    print("=" * 70)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    print("Training Data:")
    for i in range(len(X)):
        print(f"  Input: {X[i]} -> Target: {y[i][0]}")
    print()

    # Create a simple network
    network = DetailedNeuralNetwork()
    network.add_layer(Dense(2, 3))  # 2 inputs, 3 hidden neurons
    network.add_layer(Activation(Sigmoid()))
    network.add_layer(Dense(3, 1))  # 3 hidden, 1 output
    network.add_layer(Activation(Sigmoid()))
    network.set_loss(MeanSquaredError())

    # Manual training with detailed output
    print("INITIAL NETWORK STATE:")
    print("=" * 40)
    initial_pred = network.detailed_forward(X, verbose=True)
    initial_loss = network.loss_function.forward(y, initial_pred)
    print(f"Initial Loss: {initial_loss:.6f}\n")

    # Perform a few training steps with detailed analysis
    learning_rate = 0.5
    for epoch in range(3):
        print(f"TRAINING STEP {epoch + 1}")
        print("=" * 70)

        # Forward pass
        predictions = network.detailed_forward(X, verbose=(epoch < 2))
        loss = network.loss_function.forward(y, predictions)

        if epoch < 2:  # Show detailed backward pass for first two epochs
            # Backward pass
            network.detailed_backward(y, predictions, learning_rate, verbose=True)
        else:
            # Just do the backward pass without verbose output
            network.backward(y, predictions, learning_rate)

        print(f"Epoch {epoch + 1} - Loss: {loss:.6f}")
        print(f"Predictions: {predictions.ravel()}")
        print(f"Targets:     {y.ravel()}")
        print()

    # Continue training silently
    for epoch in range(100):
        predictions = network.forward(X)
        network.backward(y, predictions, learning_rate)

    # Final results
    print("FINAL RESULTS AFTER 100 EPOCHS:")
    print("=" * 40)
    final_pred = network.forward(X)
    final_loss = network.loss_function.forward(y, final_pred)

    print("Input -> Predicted | Target | Correct")
    print("-" * 35)
    for i in range(len(X)):
        pred_value = final_pred[i][0]
        target_value = y[i][0]
        correct = "✓" if abs(pred_value - target_value) < 0.5 else "✗"
        print(f"{X[i]} -> {pred_value:.4f}     | {target_value:.1f}      | {correct}")

    print(f"\nFinal Loss: {final_loss:.6f}")

    return network

def gradient_flow_visualization():
    """Visualize how gradients flow through the network"""
    print("=" * 70)
    print("GRADIENT FLOW VISUALIZATION")
    print("=" * 70)

    # Simple 1D regression problem
    X = np.array([[0.5], [1.0], [1.5]], dtype=np.float32)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)  # y = 2x

    # Small network
    network = DetailedNeuralNetwork()
    network.add_layer(Dense(1, 2))  # 1 input, 2 hidden
    network.add_layer(Activation(ReLU()))
    network.add_layer(Dense(2, 1))  # 2 hidden, 1 output
    network.set_loss(MeanSquaredError())

    print("Network Architecture:")
    print("Input(1) -> Dense(2) -> ReLU -> Dense(1) -> Output")
    print()

    # One forward and backward pass with detailed analysis
    pred = network.detailed_forward(X, verbose=True)
    network.detailed_backward(y, pred, learning_rate=0.1, verbose=True)

    # Analyze gradient magnitudes
    if network.step_by_step_info:
        gradient_info = network.step_by_step_info[-1]

        print("GRADIENT MAGNITUDE ANALYSIS:")
        print("=" * 40)
        for info in gradient_info:
            layer_num = info['layer'] + 1
            weight_grad_norm = np.linalg.norm(info['weight_gradients'])
            bias_grad_norm = np.linalg.norm(info['bias_gradients'])

            print(f"Layer {layer_num}:")
            print(f"  Weight gradient magnitude: {weight_grad_norm:.6f}")
            print(f"  Bias gradient magnitude: {bias_grad_norm:.6f}")
            print()

def activation_derivatives_demo():
    """Demonstrate how different activation functions affect gradients"""
    print("=" * 70)
    print("ACTIVATION FUNCTION DERIVATIVES DEMO")
    print("=" * 70)

    # Test inputs
    x = np.linspace(-3, 3, 7).reshape(-1, 1)

    activations = {
        'ReLU': ReLU(),
        'Sigmoid': Sigmoid()
    }

    plt.figure(figsize=(15, 5))

    for i, (name, activation) in enumerate(activations.items()):
        # Forward pass
        forward_output = activation.forward(x)
        # Derivative
        derivative = activation.backward(x)

        print(f"{name} Activation:")
        print(f"Input:      {x.ravel()}")
        print(f"Output:     {forward_output.ravel()}")
        print(f"Derivative: {derivative.ravel()}")
        print()

        # Plot
        plt.subplot(1, 3, i + 1)
        x_smooth = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_smooth = activation.forward(x_smooth)
        dy_smooth = activation.backward(x_smooth)

        plt.plot(x_smooth, y_smooth, 'b-', label=f'{name}(x)', linewidth=2)
        plt.plot(x_smooth, dy_smooth, 'r--', label=f"d{name}/dx", linewidth=2)
        plt.scatter(x, forward_output, color='blue', s=50, zorder=5)
        plt.scatter(x, derivative, color='red', s=50, zorder=5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{name} Function and Derivative')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Loss function derivatives
    plt.subplot(1, 3, 3)
    y_true = np.array([[1.0]])
    y_pred = np.linspace(0.1, 2.0, 20).reshape(-1, 1)

    mse_loss = []
    mse_derivative = []

    for pred in y_pred:
        loss = MeanSquaredError.forward(y_true, pred)
        derivative = MeanSquaredError.backward(y_true, pred)
        mse_loss.append(loss)
        mse_derivative.append(derivative[0, 0])

    plt.plot(y_pred, mse_loss, 'g-', label='MSE Loss', linewidth=2)
    plt.plot(y_pred, mse_derivative, 'm--', label='dMSE/dy_pred', linewidth=2)
    plt.axvline(x=1.0, color='k', linestyle=':', alpha=0.7, label='Target (y=1)')
    plt.xlabel('Predicted Value')
    plt.ylabel('Loss / Derivative')
    plt.title('MSE Loss and Derivative')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('activation_derivatives.png', dpi=300, bbox_inches='tight')
    plt.show()

def weight_update_tracking():
    """Track how weights change during training"""
    print("=" * 70)
    print("WEIGHT UPDATE TRACKING")
    print("=" * 70)

    # Simple dataset
    X = np.array([[0.1], [0.3], [0.7], [0.9]], dtype=np.float32)
    y = np.array([[0.2], [0.6], [1.4], [1.8]], dtype=np.float32)  # y ≈ 2x

    # Very simple network: 1 -> 1 (just linear transformation)
    network = NeuralNetwork()
    network.add_layer(Dense(1, 1))
    network.set_loss(MeanSquaredError())

    # Track weights over time
    weight_history = []
    bias_history = []
    loss_history = []

    initial_weight = network.layers[0].weights[0, 0]
    initial_bias = network.layers[0].bias[0, 0]

    print(f"Initial weight: {initial_weight:.4f}")
    print(f"Initial bias: {initial_bias:.4f}")
    print()

    learning_rate = 0.1
    for epoch in range(20):
        # Forward pass
        predictions = network.forward(X)
        loss = network.loss_function.forward(y, predictions)

        # Store current state
        current_weight = network.layers[0].weights[0, 0]
        current_bias = network.layers[0].bias[0, 0]
        weight_history.append(current_weight)
        bias_history.append(current_bias)
        loss_history.append(loss)

        # Show first few epochs in detail
        if epoch < 5:
            print(f"Epoch {epoch + 1}:")
            print(f"  Weight: {current_weight:.6f}")
            print(f"  Bias: {current_bias:.6f}")
            print(f"  Loss: {loss:.6f}")
            print(f"  Predictions: {predictions.ravel()}")
            print(f"  Targets: {y.ravel()}")

        # Backward pass
        network.backward(y, predictions, learning_rate)

        if epoch < 5:
            weight_gradient = network.layers[0].weight_gradients[0, 0]
            bias_gradient = network.layers[0].bias_gradients[0, 0]
            print(f"  Weight gradient: {weight_gradient:.6f}")
            print(f"  Bias gradient: {bias_gradient:.6f}")
            print()

    # Plot weight evolution
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(weight_history, 'b-o', markersize=4)
    plt.title('Weight Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(bias_history, 'r-o', markersize=4)
    plt.title('Bias Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Bias Value')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(loss_history, 'g-o', markersize=4)
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('weight_tracking.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Final weight: {weight_history[-1]:.6f} (started at {initial_weight:.6f})")
    print(f"Final bias: {bias_history[-1]:.6f} (started at {initial_bias:.6f})")
    print(f"Final loss: {loss_history[-1]:.6f}")

    # The optimal solution for y = 2x should have weight ≈ 2, bias ≈ 0
    print(f"Expected optimal weight: ~2.0")
    print(f"Expected optimal bias: ~0.0")

def main():
    """Run all backpropagation demonstrations"""
    print("BACKPROPAGATION DEMONSTRATIONS")
    print("=" * 70)
    print("This script provides detailed step-by-step analysis of backpropagation")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    try:
        # 1. XOR problem demonstration
        xor_network = simple_xor_demo()

        print("\n" + "="*70 + "\n")

        # 2. Gradient flow visualization
        gradient_flow_visualization()

        print("\n" + "="*70 + "\n")

        # 3. Activation derivatives demo
        activation_derivatives_demo()

        print("\n" + "="*70 + "\n")

        # 4. Weight update tracking
        weight_update_tracking()

        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED!")
        print("Check the generated PNG files for visualizations.")
        print("="*70)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
