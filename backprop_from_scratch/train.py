import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import create_simple_classifier, create_regression_network, NeuralNetwork

def generate_classification_data(n_samples=1000, n_features=2, n_classes=2, noise=0.1):
    """Generate classification dataset"""
    if n_classes == 2:
        # Binary classification - use circles dataset for interesting patterns
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        y = y.reshape(-1, 1)  # Reshape for network compatibility
    else:
        # Multi-class classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_clusters_per_class=1,
            noise=noise,
            random_state=42
        )
        # Convert to one-hot encoding
        y_onehot = np.zeros((len(y), n_classes))
        y_onehot[np.arange(len(y)), y] = 1
        y = y_onehot

    return X, y

def generate_regression_data(n_samples=1000, n_features=1, noise=0.1):
    """Generate regression dataset"""
    if n_features == 1:
        # Simple sinusoidal function
        X = np.linspace(-2*np.pi, 2*np.pi, n_samples).reshape(-1, 1)
        y = np.sin(X) + np.random.normal(0, noise, X.shape)
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=42
        )
        y = y.reshape(-1, 1)

    return X, y

def plot_training_history(history, title="Training History"):
    """Plot training loss and accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    axes[0].plot(history['loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # Plot accuracy
    axes[1].plot(history['accuracy'])
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_classification_results(X, y, network, title="Classification Results"):
    """Plot classification results and decision boundary"""
    # Create a mesh to plot the decision boundary
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = network.predict(mesh_points)

    if Z.shape[1] == 1:  # Binary classification
        Z = (Z > 0.5).astype(int)
    else:  # Multi-class
        Z = np.argmax(Z, axis=1)

    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)

    if y.shape[1] == 1:  # Binary classification
        colors = ['red', 'blue']
        for i, color in enumerate(colors):
            idx = np.where(y.ravel() == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, s=30, alpha=0.7,
                       label=f'Class {i}', edgecolors='black', linewidth=0.5)
    else:  # Multi-class
        y_labels = np.argmax(y, axis=1)
        unique_labels = np.unique(y_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, color in zip(unique_labels, colors):
            idx = np.where(y_labels == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=[color], s=30, alpha=0.7,
                       label=f'Class {i}', edgecolors='black', linewidth=0.5)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_regression_results(X, y, network, X_test, y_test, title="Regression Results"):
    """Plot regression results"""
    plt.figure(figsize=(12, 4))

    # Sort for better plotting
    if X.shape[1] == 1:
        sort_idx = np.argsort(X.ravel())
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        predictions = network.predict(X_sorted)

        plt.subplot(1, 2, 1)
        plt.scatter(X, y, alpha=0.6, label='Training Data', s=20)
        plt.plot(X_sorted, predictions, 'r-', linewidth=2, label='Network Prediction')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Training Data vs Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Test predictions
        test_predictions = network.predict(X_test)

        plt.subplot(1, 2, 2)
        plt.scatter(X_test, y_test, alpha=0.6, label='Test Data', s=20)
        plt.scatter(X_test, test_predictions, alpha=0.6, label='Predictions', s=20, color='red')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Test Data vs Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        # Multi-dimensional input - just show loss comparison
        train_pred = network.predict(X)
        test_pred = network.predict(X_test)

        plt.subplot(1, 2, 1)
        plt.scatter(y, train_pred, alpha=0.6, s=20)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Training: True vs Predicted')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(y_test, test_pred, alpha=0.6, s=20)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Test: True vs Predicted')
        plt.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_binary_classification():
    """Train a binary classification network"""
    print("=" * 60)
    print("BINARY CLASSIFICATION DEMO")
    print("=" * 60)

    # Generate data
    X, y = generate_classification_data(n_samples=1000, n_classes=2, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Create and train network
    network = create_simple_classifier(input_size=2, hidden_size=10, output_size=1, activation='relu')

    print("\nTraining binary classification network...")
    history = network.train(X_train, y_train, epochs=500, learning_rate=0.1, batch_size=32, verbose=True)

    # Evaluate
    train_pred = network.predict(X_train)
    test_pred = network.predict(X_test)

    train_acc = np.mean((train_pred > 0.5) == y_train)
    test_acc = np.mean((test_pred > 0.5) == y_test)

    print(f"\nFinal Results:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Plot results
    plot_training_history(history, "Binary Classification Training")
    plot_classification_results(X_train, y_train, network, "Binary Classification Decision Boundary")

    return network, history

def train_regression():
    """Train a regression network"""
    print("=" * 60)
    print("REGRESSION DEMO")
    print("=" * 60)

    # Generate data
    X, y = generate_regression_data(n_samples=500, n_features=1, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")

    # Create and train network
    network = create_regression_network(input_size=1, hidden_sizes=[20, 20], output_size=1)

    print("\nTraining regression network...")
    history = network.train(X_train, y_train, epochs=1000, learning_rate=0.01, batch_size=16, verbose=True)

    # Evaluate
    train_pred = network.predict(X_train)
    test_pred = network.predict(X_test)

    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)

    print(f"\nFinal Results:")
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")

    # Plot results
    plot_training_history(history, "Regression Training")
    plot_regression_results(X_train, y_train, network, X_test, y_test, "Regression Results")

    return network, history

def compare_activations():
    """Compare different activation functions"""
    print("=" * 60)
    print("ACTIVATION FUNCTION COMPARISON")
    print("=" * 60)

    # Generate data
    X, y = generate_classification_data(n_samples=800, n_classes=2, noise=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    activations = ['relu', 'sigmoid', 'tanh']
    results = {}

    plt.figure(figsize=(15, 5))

    for i, activation in enumerate(activations):
        print(f"\nTraining with {activation.upper()} activation...")

        # Create network
        network = create_simple_classifier(input_size=2, hidden_size=15, output_size=1, activation=activation)

        # Train
        history = network.train(X_train, y_train, epochs=300, learning_rate=0.1, batch_size=32, verbose=False)

        # Evaluate
        test_pred = network.predict(X_test)
        test_acc = np.mean((test_pred > 0.5) == y_test)

        results[activation] = {
            'accuracy': test_acc,
            'history': history,
            'network': network
        }

        print(f"{activation.upper()} - Test Accuracy: {test_acc:.4f}")

        # Plot training curves
        plt.subplot(1, 3, i + 1)
        plt.plot(history['loss'], label='Loss')
        plt.plot(history['accuracy'], label='Accuracy')
        plt.title(f'{activation.upper()} Activation')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('activation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results

def main():
    """Main function to run all demos"""
    # Set random seed for reproducibility
    np.random.seed(42)

    print("BACKPROPAGATION FROM SCRATCH - NEURAL NETWORK DEMO")
    print("=" * 70)

    try:
        # Binary classification demo
        binary_network, binary_history = train_binary_classification()

        print("\n" + "="*70 + "\n")

        # Regression demo
        regression_network, regression_history = train_regression()

        print("\n" + "="*70 + "\n")

        # Activation function comparison
        activation_results = compare_activations()

        print("\n" + "="*70)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("Check the generated PNG files for visualizations.")
        print("="*70)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
