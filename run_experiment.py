import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from functions.evaluate import evaluate
from functions.train import train
from models.Classifier import Classifier
from models.HypernetClassifier import HyperNetClassifier
from utils.MNIST_loaders import MNIST_loaders


def write_params_to_file(params: dict, choosen_param: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "params.json")
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)

def generate_chart(baseline_values: list, hypernet_values: list, title: str, save_dir: str):
    epochs = range(1, len(baseline_values) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, baseline_values, label='Baseline Classifier')
    plt.plot(epochs, hypernet_values, label='Hypernet Classifier')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filepath)
    plt.close()

def generate_accuracy_chart(baseline_train_accuracy: list, hypernet_train_accuracy: list, baseline_val_accuracy: list, hypernet_val_accuracy: list, title: str, save_dir: str):
    epochs = range(1, len(baseline_train_accuracy) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, baseline_train_accuracy, label='Baseline Train Accuracy')
    plt.plot(epochs, baseline_val_accuracy, label='Baseline Validation Accuracy')
    plt.plot(epochs, hypernet_train_accuracy, label='Hypernet Train Accuracy')
    plt.plot(epochs, hypernet_val_accuracy, label='Hypernet Validation Accuracy')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
    plt.savefig(filepath)
    plt.close()

def find_list_of_params(params: dict, list_type_params: list = []):
    for param in params.keys():
        if (param not in list_type_params and type(params[param]) == list) \
            or (param in list_type_params and type(params[param][0]) == list):
            return param, params[param]
            
    return None, None

def run_train_and_eval(params: dict, device: torch.device, choosen_param: str = None):
    model = Classifier(params["input_size"], params["output_size"], zeroed_weights_fraction=params["zeroed_weights_in_baseline"])
    hypernet_model = HyperNetClassifier(params["input_size"], params["output_size"], hidden_sizes=params["hidden_size"], device=device)

    if params["optimizer"] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        hypernet_optimizer = optim.SGD(hypernet_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        hypernet_optimizer = optim.Adam(hypernet_model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
        
    train_loader, val_loader, _ = MNIST_loaders()
    
    train_loss, h_train_loss, train_accuracy, h_train_accuracy = [], [], [], []
    validation_accuracy, h_validation_accuracy, validation_ece, h_validation_ece = [], [], [], []

    for epoch in range(1, params["num_epochs"] + 1):
        avg_loss, avg_hypernet_loss, accuracy, hypernet_accuracy = train(model, hypernet_model, train_loader, optimizer, hypernet_optimizer, device=device)
        val_accuracy, ece = evaluate(model, val_loader, n_bins=params["n_bins"], device=device)
        hypernet_val_accuracy, hypernet_ece = evaluate(hypernet_model, val_loader, device=device)
        
        train_loss.append(avg_loss)
        h_train_loss.append(avg_hypernet_loss)
        train_accuracy.append(accuracy)
        h_train_accuracy.append(hypernet_accuracy)
        
        validation_accuracy.append(val_accuracy)
        h_validation_accuracy.append(hypernet_val_accuracy)
        validation_ece.append(ece)
        h_validation_ece.append(hypernet_ece)
        
    save_dir = f"results/{choosen_param}_{params[choosen_param]} - {datetime.now().strftime('%d-%m-%Y %H-%M-%S')}"
    
    write_params_to_file(params, choosen_param, save_dir=save_dir)
    generate_chart(train_loss, h_train_loss, title="Train Loss", save_dir=save_dir)
    
    generate_chart(validation_ece, h_validation_ece, title="Validation ECE", save_dir=save_dir)
    generate_accuracy_chart(train_accuracy, h_train_accuracy, validation_accuracy, h_validation_accuracy, title="Train and Val Accuracy", save_dir=save_dir)
    
# --- Hyperparameters (things you can easily change!) ---
params = {
    "num_epochs": 20,
    "batch_size": 64,
    "weight_decay": 0.001,
    "validation_split": 0.2,  # Part of the training data to use for validation
    "random_seed": 42,
    "n_bins": 10, # Number of bins for calculating ECE
    "input_size": 28 * 28,
    "output_size": 10,

    "learning_rate": 0.0001,
    "optimizer": 'SGD',
    "hidden_size": [256, 64, 256],
    "zeroed_weights_in_baseline": 0.5,
    "hypernet_ensemble_num": 3,
}

torch.manual_seed(params["random_seed"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

list_type_params = ["hidden_size"]

choosen_param, param_list = find_list_of_params(params, list_type_params)

if param_list is None:
    print("No list of parameters found...")
    run_train_and_eval(params, device, "num_epochs")
else:
    for param_value in param_list:
        params[choosen_param] = param_value
        
        print(f"Running experiment with {choosen_param}: {param_value}")
        run_train_and_eval(params, device, choosen_param)
        
    
    

