import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Arrow
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
from model import create_simple_classifier, Dense, Activation, ReLU, Sigmoid, Tanh, MeanSquaredError
import networkx as nx

def visualize_network_architecture(network, title="Neural Network Architecture"):
    """Visualize the network architecture"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Count layers and neurons
    layer_info = []
    for layer in network.layers:
        if isinstance(layer, Dense):
            layer_info.append({
                'type': 'Dense',
                'input_size': layer.weights.shape[0],
                'output_size': layer.weights.shape[1]
            })
        elif isinstance(layer, Activation):
            layer_info.append({
                'type': 'Activation',
                'activation': type(layer.activation).__name__
            })

    # Calculate positions
    max_neurons = max([info.get('output_size', info.get('input_size', 1)) for info in layer_info if 'output_size' in info] + [layer_info[0]['input_size']])
    layer_positions = np.linspace(0.1, 0.9, len([info for info in layer_info if info['type'] == 'Dense']) + 1)

    colors = {'Dense': 'lightblue', 'Activation': 'lightcoral'}
    neuron_positions = {}

    # Draw input layer
    input_size = layer_info[0]['input_size']
    y_positions = np.linspace(0.2, 0.8, input_size)
    for i, y in enumerate(y_positions):
        circle = Circle((layer_positions[0], y), 0.03, color='lightgreen', ec='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(layer_positions[0], y, f'x{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
    neuron_positions[0] = list(zip([layer_positions[0]] * input_size, y_positions))

    # Draw hidden and output layers
    layer_idx = 1
    for i, info in enumerate(layer_info):
        if info['type'] == 'Dense':
            output_size = info['output_size']
            y_positions = np.linspace(0.2, 0.8, output_size)

            # Draw neurons
            for j, y in enumerate(y_positions):
                if layer_idx == len(layer_positions) - 1:  # Output layer
                    circle = Circle((layer_positions[layer_idx], y), 0.03, color='orange', ec='black', linewidth=1.5)
                    ax.text(layer_positions[layer_idx], y, f'y{j+1}', ha='center', va='center', fontsize=10, fontweight='bold')
                else:  # Hidden layer
                    circle = Circle((layer_positions[layer_idx], y), 0.03, color='lightblue', ec='black', linewidth=1.5)
                    ax.text(layer_positions[layer_idx], y, f'h{j+1}', ha='center', va='center', fontsize=10, fontweight='bold')
                ax.add_patch(circle)

            # Draw connections
            if layer_idx > 0:
                prev_positions = neuron_positions[layer_idx - 1]
                curr_positions = list(zip([layer_positions[layer_idx]] * output_size, y_positions))

                for prev_x, prev_y in prev_positions:
                    for curr_x, curr_y in curr_positions:
                        ax.plot([prev_x, curr_x], [prev_y, curr_y], 'gray', alpha=0.5, linewidth=1)

            neuron_positions[layer_idx] = list(zip([layer_positions[layer_idx]] * output_size, y_positions))
            layer_idx += 1

    # Add layer labels
    layer_idx = 0
    for i, info in enumerate(layer_info):
        if info['type'] == 'Dense':
            if layer_idx == 0:
                ax.text(layer_positions[layer_idx], 0.1, 'Input Layer', ha='center', va='center',
                       fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
            elif layer_idx == len(layer_positions) - 1:
                ax.text(layer_positions[layer_idx], 0.1, 'Output Layer', ha='center', va='center',
                       fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.7))
            else:
                ax.text(layer_positions[layer_idx], 0.1, f'Hidden Layer {layer_idx}', ha='center', va='center',
                       fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            layer_idx += 1
        elif info['type'] == 'Activation':
            # Add activation function label between layers
            if layer_idx > 0:
                x_pos = (layer_positions[layer_idx-1] + layer_positions[layer_idx]) / 2
                ax.text(x_pos, 0.05, info['activation'], ha='center', va='center',
                       fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.7))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_gradient_flow(network, X, y, learning_rate=0.1):
    """Visualize how gradients flow through the network"""
    # Forward pass
    predictions = network.forward(X)
    loss = network.loss_function.forward(y, predictions)

    # Store original weights
    original_weights = []
    for layer in network.layers:
        if isinstance(layer, Dense):
            original_weights.append((layer.weights.copy(), layer.bias.copy()))

    # Backward pass
    gradient = network.loss_function.backward(y, predictions)

    # Track gradients
    gradient_magnitudes = []
    weight_updates = []

    for layer in reversed(network.layers):
        if isinstance(layer, Dense):
            # Store current state
            old_weights = layer.weights.copy()
            old_bias = layer.bias.copy()

        # Backward pass
        gradient = layer.backward(gradient, learning_rate)

        if isinstance(layer, Dense):
            # Calculate magnitudes
            weight_grad_mag = np.linalg.norm(layer.weight_gradients)
            bias_grad_mag = np.linalg.norm(layer.bias_gradients)
            weight_update_mag = np.linalg.norm(layer.weights - old_weights)
            bias_update_mag = np.linalg.norm(layer.bias - old_bias)

            gradient_magnitudes.append(weight_grad_mag)
            weight_updates.append(weight_update_mag)

    # Reverse to match forward order
    gradient_magnitudes = gradient_magnitudes[::-1]
    weight_updates = weight_updates[::-1]

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss landscape (for 2D input)
    if X.shape[1] == 2:
        axes[0].set_title('Loss Landscape & Gradient Direction')
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = []
        for point in mesh_points:
            pred = network.forward(point.reshape(1, -1))
            sample_loss = network.loss_function.forward(y[:1], pred)  # Use first target
            Z.append(sample_loss)

        Z = np.array(Z).reshape(xx.shape)
        im = axes[0].contour(xx, yy, Z, levels=20, alpha=0.6)
        axes[0].scatter(X[:, 0], X[:, 1], c='red', s=100, alpha=0.8, marker='x', linewidths=3)
        axes[0].set_xlabel('Feature 1')
        axes[0].set_ylabel('Feature 2')
    else:
        axes[0].text(0.5, 0.5, 'Gradient Flow\n(Input dim > 2)', ha='center', va='center',
                     transform=axes[0].transAxes, fontsize=14)
        axes[0].axis('off')

    # Gradient magnitudes by layer
    layer_names = [f'Layer {i+1}' for i in range(len(gradient_magnitudes))]
    bars = axes[1].bar(layer_names, gradient_magnitudes, color='skyblue', alpha=0.7, edgecolor='navy')
    axes[1].set_title('Gradient Magnitudes by Layer')
    axes[1].set_ylabel('Gradient Magnitude')
    axes[1].tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, mag in zip(bars, gradient_magnitudes):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                     f'{mag:.4f}', ha='center', va='bottom', fontsize=10)

    # Weight update magnitudes
    bars = axes[2].bar(layer_names, weight_updates, color='lightcoral', alpha=0.7, edgecolor='darkred')
    axes[2].set_title('Weight Update Magnitudes by Layer')
    axes[2].set_ylabel('Update Magnitude')
    axes[2].tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, update in zip(bars, weight_updates):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                     f'{update:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_activation_functions():
    """Visualize different activation functions and their derivatives"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    x = np.linspace(-5, 5, 100)
    activations = {
        'ReLU': ReLU(),
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh()
    }

    colors = ['blue', 'green', 'red']

    for i, (name, activation) in enumerate(activations.items()):
        # Function plot
        y = activation.forward(x.reshape(-1, 1)).ravel()
        axes[0, i].plot(x, y, color=colors[i], linewidth=3, label=f'{name}(x)')
        axes[0, i].set_title(f'{name} Activation Function', fontsize=14, fontweight='bold')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel(f'{name}(x)')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].legend()

        # Derivative plot
        dy = activation.backward(x.reshape(-1, 1)).ravel()
        axes[1, i].plot(x, dy, color=colors[i], linewidth=3, linestyle='--', label=f"d{name}/dx")
        axes[1, i].set_title(f'{name} Derivative', fontsize=14, fontweight='bold')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel(f"d{name}/dx")
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()

        # Add some analysis text
        if name == 'ReLU':
            axes[1, i].text(0.05, 0.95, 'Derivative = 1 for x > 0\nDerivative = 0 for x ≤ 0\n(Vanishing gradient for x ≤ 0)',
                           transform=axes[1, i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
        elif name == 'Sigmoid':
            axes[1, i].text(0.05, 0.95, 'Maximum derivative = 0.25\n(Vanishing gradient problem\nfor large |x|)',
                           transform=axes[1, i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
        else:  # Tanh
            axes[1, i].text(0.05, 0.95, 'Maximum derivative = 1\n(Better than sigmoid\nbut still saturates)',
                           transform=axes[1, i].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_loss_landscape_2d():
    """Visualize loss landscape for a simple 2D problem"""
    # Create simple dataset
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)  # XOR

    # Create network
    network = create_simple_classifier(input_size=2, hidden_size=2, output_size=1, activation='sigmoid')

    # Get initial weights (we'll use first layer weights for visualization)
    w1_range = np.linspace(-3, 3, 50)
    w2_range = np.linspace(-3, 3, 50)
    W1, W2 = np.meshgrid(w1_range, w2_range)

    loss_surface = np.zeros_like(W1)

    # Calculate loss for each weight combination
    original_weights = network.layers[0].weights.copy()

    for i in range(len(w1_range)):
        for j in range(len(w2_range)):
            # Set new weights (only modify first two weights for visualization)
            network.layers[0].weights[0, 0] = W1[j, i]
            network.layers[0].weights[1, 0] = W2[j, i]

            # Calculate loss
            pred = network.forward(X)
            loss = network.loss_function.forward(y, pred)
            loss_surface[j, i] = loss

    # Restore original weights
    network.layers[0].weights = original_weights

    # Plot
    fig = plt.figure(figsize=(15, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(W1, W2, loss_surface, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Weight 1')
    ax1.set_ylabel('Weight 2')
    ax1.set_zlabel('Loss')
    ax1.set_title('3D Loss Surface')

    # 2D contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(W1, W2, loss_surface, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('Weight 1')
    ax2.set_ylabel('Weight 2')
    ax2.set_title('Loss Contours')
    ax2.grid(True, alpha=0.3)

    # Add gradient descent path visualization
    ax3 = fig.add_subplot(133)
    im = ax3.contourf(W1, W2, loss_surface, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(im, ax=ax3, shrink=0.8)

    # Simulate gradient descent path
    network.layers[0].weights = original_weights.copy()
    weight_path = []

    for epoch in range(20):
        w1_current = network.layers[0].weights[0, 0]
        w2_current = network.layers[0].weights[1, 0]
        weight_path.append((w1_current, w2_current))

        # Training step
        pred = network.forward(X)
        network.backward(y, pred, learning_rate=1.0)

    # Plot path
    if len(weight_path) > 1:
        path_w1, path_w2 = zip(*weight_path)
        ax3.plot(path_w1, path_w2, 'ro-', linewidth=2, markersize=6,
                label='Gradient Descent Path', markerfacecolor='white', markeredgewidth=2)
        ax3.plot(path_w1[0], path_w2[0], 'gs', markersize=10, label='Start')
        ax3.plot(path_w1[-1], path_w2[-1], 'rs', markersize=10, label='End')

    ax3.set_xlabel('Weight 1')
    ax3.set_ylabel('Weight 2')
    ax3.set_title('Gradient Descent Path')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('loss_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_learning_dynamics():
    """Visualize learning dynamics with different learning rates"""
    # Simple linear regression problem
    np.random.seed(42)
    X = np.linspace(0, 1, 20).reshape(-1, 1)
    y = (2 * X + 0.5 + 0.1 * np.random.randn(20, 1))  # y = 2x + 0.5 + noise

    learning_rates = [0.01, 0.1, 0.5, 2.0]
    colors = ['blue', 'green', 'orange', 'red']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (lr, color) in enumerate(zip(learning_rates, colors)):
        # Create simple network
        from model import NeuralNetwork, Dense, MeanSquaredError
        network = NeuralNetwork()
        network.add_layer(Dense(1, 1))  # Simple linear layer
        network.set_loss(MeanSquaredError())

        # Track training
        losses = []
        weights = []
        biases = []

        for epoch in range(100):
            pred = network.forward(X)
            loss = network.loss_function.forward(y, pred)
            losses.append(loss)
            weights.append(network.layers[0].weights[0, 0])
            biases.append(network.layers[0].bias[0, 0])

            network.backward(y, pred, lr)

        # Plot results
        ax = axes[idx]
        ax2 = ax.twinx()

        # Loss curve
        line1 = ax.plot(losses, color=color, linewidth=2, label='Loss')
        ax.set_ylabel('Loss', color=color)
        ax.tick_params(axis='y', labelcolor=color)

        # Weight evolution
        line2 = ax2.plot(weights, color='gray', linestyle='--', alpha=0.7, label='Weight')
        ax2.set_ylabel('Weight Value', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        ax.set_xlabel('Epoch')
        ax.set_title(f'Learning Rate = {lr}')
        ax.grid(True, alpha=0.3)

        # Add final values as text
        final_weight = weights[-1]
        final_loss = losses[-1]
        ax.text(0.05, 0.95, f'Final Weight: {final_weight:.3f}\nFinal Loss: {final_loss:.6f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Check for divergence
        if final_loss > losses[0] * 2:
            ax.text(0.5, 0.5, 'DIVERGED!', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16, color='red', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('learning_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run all visualizations"""
    print("BACKPROPAGATION VISUALIZATIONS")
    print("=" * 70)

    # Set random seed
    np.random.seed(42)

    try:
        # Create a sample network for visualization
        network = create_simple_classifier(input_size=2, hidden_size=4, output_size=1, activation='relu')

        # 1. Network architecture
        print("1. Visualizing network architecture...")
        visualize_network_architecture(network, "Sample Neural Network Architecture")

        # 2. Activation functions
        print("2. Visualizing activation functions...")
        visualize_activation_functions()

        # 3. Loss landscape
        print("3. Visualizing loss landscape...")
        visualize_loss_landscape_2d()

        # 4. Learning dynamics
        print("4. Visualizing learning dynamics...")
        visualize_learning_dynamics()

        # 5. Gradient flow (with sample data)
        print("5. Visualizing gradient flow...")
        X_sample = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        y_sample = np.array([[1], [0], [1]])
        visualize_gradient_flow(network, X_sample, y_sample)

        print("\n" + "="*70)
        print("ALL VISUALIZATIONS COMPLETED!")
        print("Generated files:")
        print("- network_architecture.png")
        print("- activation_functions.png")
        print("- loss_landscape.png")
        print("- learning_dynamics.png")
        print("- gradient_flow.png")
        print("="*70)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
