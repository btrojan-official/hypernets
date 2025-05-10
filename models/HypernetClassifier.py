import torch
import torch.nn as nn

class HyperNetClassifier(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list=[2400, 1200, 2400], device: str="cpu", ensemble_num: int=1, use_previous_weights: bool=False):
        super(HyperNetClassifier, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.num_weights = input_size * output_size
        
        self.input_layer = nn.Linear(self.num_weights, hidden_sizes[0], bias=False).to(device)
        self.hidden_1 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(device)
        self.hidden_2 = nn.Linear(hidden_sizes[1], hidden_sizes[2]).to(device)
        self.output_layer = nn.Linear(hidden_sizes[2], self.num_weights, bias=False).to(device)
        
        self.device = device
        self.ensemble_num = ensemble_num
        
        self.use_previous_weights = use_previous_weights
        self.previous_weights = None

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 28 * 28)
        
        if self.training:
            ensemble_num = 1
        else:
            ensemble_num = self.ensemble_num
        
        outputs = []
        
        for _ in range(ensemble_num):
            mask = (torch.rand(self.num_weights, requires_grad=False) >= 0.5).to(self.device)
            hypernet_input = torch.randn(self.num_weights, requires_grad=True).to(self.device)
            if self.use_previous_weights and self.previous_weights != None:
                hypernet_input[mask] = self.previous_weights[mask]
            else:
                hypernet_input[~mask] = torch.zeros(self.num_weights, requires_grad=True).to(self.device)[~mask]
            
            hypernet_output = torch.relu(self.input_layer(hypernet_input.clone()))
            hypernet_output = torch.relu(self.hidden_1(hypernet_output))
            hypernet_output = torch.relu(self.hidden_2(hypernet_output))
            hypernet_output = self.output_layer(hypernet_output)

            hypernet_output[mask] = hypernet_input[mask] #* It copies weigths from previous iteration or from random initialization if there is no previous iteration
            
            if self.use_previous_weights:
                self.previous_weights = hypernet_output.clone().detach()
            
            weights = torch.reshape(hypernet_output, (self.input_size, self.output_size))
            
            outputs.append(x @ weights)
        
        if self.training:
            return outputs[0]
        else:
            softmaxed_outputs = torch.softmax(torch.stack(outputs, dim=0), dim=2)
            return torch.mean(softmaxed_outputs, dim=0)