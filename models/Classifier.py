import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout=0):
        super(Classifier, self).__init__()
        # This is a single linear layer (like a simple connection of all inputs to all outputs)
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # The input images are 28x28 pixels, so we need to flatten them into a single vector of 784 elements
        x = x.view(-1, 28 * 28)
        # Pass the flattened vector through the linear layer
        x = self.linear(self.dropout(x))
        return x