import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, zeroed_weights_fraction=0):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        self.zeroed_weights_fraction = zeroed_weights_fraction
        
        self.zeroed_weight = None
        self.zeroed_weight_indices = None

    def zero_out_weights(self, zeroed_weights):
        with torch.no_grad(): 
            weight = self.linear.weight
            num_elements = weight.numel()
            num_to_zero = int(num_elements * zeroed_weights)

            indices = torch.randperm(num_elements)[:num_to_zero]
            flat_weight = weight.view(-1)
            
            self.zeroed_weight_indices = indices.clone()
            self.zeroed_weight = flat_weight[indices].clone()
            
            flat_weight[indices] = 0.0
            
    def unzero_out_weights(self):
        with torch.no_grad():
            if self.zeroed_weight_indices != None and self.zeroed_weight != None:
                flat_weight = self.linear.weight.view(-1)
                flat_weight[self.zeroed_weight_indices] = self.zeroed_weight.clone()
                
                self.zeroed_weight_indices = None
                self.zeroed_weight = None

    def forward(self, x):
        self.unzero_out_weights()
        self.zero_out_weights(self.zeroed_weights_fraction)

        x = x.view(-1, 28 * 28)
        x = self.linear(x)
        
        return x