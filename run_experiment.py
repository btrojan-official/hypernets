def find_list_of_params(params: dict, list_type_params: list = []):
    choosen_params = []

    for param in params.keys():
        if (param not in list_type_params and type(param) == list) \
            or (param in list_type_params and type(param[0]) == list):
            choosen_params.append(param)
# --- Hyperparameters (things you can easily change!) ---
params = {
    "num_epochs": 5,
    "batch_size": 64,
    "weight_decay": 0.001,
    "validation_split": 0.2,  # Part of the training data to use for validation
    "random_seed": 42,
    "n_bins": 10, # Number of bins for calculating ECE
    "input_size": 28 * 28,
    "output_size": 10,

    "learning_rate": 0.0001,
    "optimizer": 'Adam',
    "hidden_size": [256, 64, 256],
    "zeroed_weights_in_baseline": 0.5,
    "hypernet_ensemble_num": 3,
}
list_type_params = ["hidden_size"]

