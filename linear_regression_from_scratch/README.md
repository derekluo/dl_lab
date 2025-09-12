# Linear Regression From Scratch

Simple implementation of linear regression (`y = Wx + B`) using PyTorch, demonstrating:
- Forward pass, loss calculation, backpropagation, gradient descent
- Training on synthetic data with noise
- Model evaluation with metrics and visualization

## Architecture
- **Model**: `y_pred = W * X + B` (single linear layer)  
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Stochastic Gradient Descent (SGD)

## Files
- `model.py` - LinearRegressionModel class
- `generate_data.py` - Create synthetic dataset
- `train.py` - Train the model  
- `inference.py` - Evaluate trained model
- `data/` - Generated datasets and results

## Usage

### Quick Start
```bash
cd linear_from_scratch
python generate_data.py --num_samples 1000
python train.py --epochs 100
python inference.py
```

### Parameters
- **generate_data.py**: `--num_samples 1000 --train_split 0.8`
- **train.py**: `--learning_rate 0.01 --epochs 100`
- **inference.py**: `--data_dir data`

## Data Format
CSV files with columns: `X` (features), `y` (noisy targets), `y_true` (ground truth)
- True parameters: W=2.0, B=1.0
- Relationship: y = 2x + 1 + noise

## Outputs
- **Training**: Loss curve plot, saved model weights
- **Evaluation**: Test metrics (MSE, MAE, R²), predictions visualization

## Dependencies
```bash
pip install torch matplotlib numpy pandas
```

## Learning Goals
- Understand ML training loop fundamentals
- Practice PyTorch tensor operations and autograd
- Learn model evaluation and visualization techniques