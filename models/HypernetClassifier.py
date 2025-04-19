import torch
import torch.nn as nn

class HyperNetClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[2400, 1200, 2400], device="cpu"):
        super(HyperNetClassifier, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.num_weights = input_size * output_size
        
        self.input_layer = nn.Linear(self.num_weights, hidden_sizes[0], bias=False)
        self.hidden_1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.hidden_2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.output_layer = nn.Linear(hidden_sizes[2], self.num_weights, bias=False)
        
        self.device = device

    def forward(self, x):
        x = x.view(-1, 28 * 28)

        mask = (torch.rand(self.num_weights, requires_grad=False) >= 0.5).int().to(self.device)
        hypernet_input = torch.randn(self.num_weights, requires_grad=True).to(self.device) * mask
        
        hypernet_output = torch.relu(self.input_layer(hypernet_input.clone()))
        hypernet_output = torch.relu(self.hidden_1(hypernet_output))
        hypernet_output = torch.relu(self.hidden_2(hypernet_output))
        hypernet_output = self.output_layer(hypernet_output)

        hypernet_output[mask] = hypernet_input[mask]
        weights = torch.reshape(hypernet_output, (self.input_size, self.output_size))

        return x @ weights