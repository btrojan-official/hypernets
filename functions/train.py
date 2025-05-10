import torch
import torch.nn as nn

def train(model: nn.Module, hypernet_model: nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, hypernet_optimizer: torch.optim.Optimizer, device: str="cpu"):
    model.train()  # Set the model to training mode
    hypernet_model.train()
    
    model = model.to(device)
    hypernet_model = hypernet_model.to(device)
    
    total_loss = 0
    correct = 0
    
    hypernet_total_loss = 0
    hypernet_correct = 0
    
    total = 0
    
    for _, (data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        hypernet_optimizer.zero_grad()
        
        data = data.to(device)
        target = target.to(device)
        
        output = model(data)
        hypernet_output = hypernet_model(data)

        loss = nn.CrossEntropyLoss()(output, target)
        hypernet_loss = nn.CrossEntropyLoss()(hypernet_output, target)

        loss.backward()
        hypernet_loss.backward()

        optimizer.step()
        hypernet_optimizer.step()

        total_loss += loss.item()
        hypernet_total_loss += hypernet_loss.item()
        
        _, predicted = torch.max(output.data, 1)
        _, hypernet_predicted = torch.max(hypernet_output.data, 1)
        
        total += target.size(0)
        
        correct += (predicted == target).sum().item()
        hypernet_correct += (hypernet_predicted == target).sum().item()

        # if (batch_idx + 1) % 100 == 0:
        #     print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
        #           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}. Hypernet Loss: {hypernet_loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    avg_hypernet_loss = hypernet_total_loss / len(train_loader)
    
    accuracy = 100. * correct / total
    hypernet_accuracy = 100. * hypernet_correct / total
    
    return avg_loss, avg_hypernet_loss, accuracy, hypernet_accuracy