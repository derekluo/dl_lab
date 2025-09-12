import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Defines a single linear layer.
        # input_features=1, output_features=1
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        # This is where the input tensor `x` is passed through the layer.
        return self.linear(x)
