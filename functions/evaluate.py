import torch
import torch.nn as nn

from models.HypernetClassifier import HyperNetClassifier

def evaluate(model: nn.Module, data_loader: torch.utils.data.DataLoader, n_bins: int=10, device: str="cpu"):
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    # For ECE calculation
    bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_corrects = torch.zeros(n_bins).to(device)
    bin_totals = torch.zeros(n_bins).to(device)
    bin_confidences = torch.zeros(n_bins).to(device)

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Calculate probabilities and confidences for ECE
            if type(model) != HyperNetClassifier:
                probabilities = torch.softmax(output, dim=1)
            else:
                probabilities = output
            confidences, predictions = torch.max(probabilities, dim=1)
            accuracies = predictions.eq(target)

            for i in range(n_bins):
                in_bin = (confidences >= bin_lowers[i]) & (confidences < bin_uppers[i])
                bin_totals[i] += in_bin.sum().item()
                bin_corrects[i] += (accuracies[in_bin]).sum().item()
                bin_confidences[i] += (confidences[in_bin]).sum().item()

    accuracy = 100. * correct / total

    # Calculate ECE
    ece = torch.zeros(1, device=device)
    for i in range(n_bins):
        if bin_totals[i] > 0:
            bin_accuracy = bin_corrects[i] / bin_totals[i]
            avg_confidence = bin_confidences[i] / bin_totals[i]
            ece += torch.abs(avg_confidence - bin_accuracy) * bin_totals[i]
    ece = ece / total
    ece = ece.item()

    return accuracy, ece