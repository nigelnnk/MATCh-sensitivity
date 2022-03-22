import pandas as pd
import torch
import sys
import logging
import datetime

import ray
from ray import tune
from ray.tune import schedulers

from src.featurization.data_utils import load_data_from_df, construct_loader
from src.transformer import make_model
from argparser import make_args

# -----------------------------------------Train model---------------------------------------
def train_model(model, train_data_loader, epoch, learning_rate):
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epoch)
    for e in range(epoch):
        for batch in train_data_loader:
            optimiser.zero_grad()
            adjacency_matrix, node_features, distance_matrix, y = batch
            adjacency_matrix = adjacency_matrix.to(device)
            node_features = node_features.to(device)
            distance_matrix = distance_matrix.to(device)
            y = y.to(device)

            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            input_data = (node_features, batch_mask, adjacency_matrix, distance_matrix, None)
            output = model(input_data)
            loss = loss_fn(output, y)
            loss.backward()
            optimiser.step()

        scheduler.step()
        logging.debug(f"Training Epoch: {e}\tLoss: {loss.item()}")


# ------------------------------------------Test model---------------------------------------
def test_model(model, test_X, test_y):
    model.eval()
    test_loss = 0
    num_data = len(test_X)
    smiles_list = [elem[3] for elem in test_X]
    test_data_loader = construct_loader(test_X, test_y, num_data, shuffle=False)

    with torch.no_grad():
        for batch in test_data_loader:
            adjacency_matrix, node_features, distance_matrix, y = batch
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            output = model((node_features, batch_mask, adjacency_matrix, distance_matrix, None))
            test_loss += loss_fn(output, y).item()
            logging.info(f"SMILES:{smiles_list}\nOUTPUT:{output.squeeze().tolist()}")
            logging.info(f"y:{y.squeeze().tolist()}")
    print(f"Sum loss of {test_loss} over {num_data} samples")
    return test_loss


def cv(config, checkpoint_dir=None):
    num_data = len(X)
    num_test = int(num_data/cv_fold)
    total_loss = 0
    for i in range(cv_fold):
        if i == cv_fold - 1:
            test_X, test_y = X[i*num_test:], y[i*num_test:]
            train_X = X[:i*num_test]
            train_y = y[:i*num_test]
        else:
            test_X, test_y = X[i*num_test:(i+1)*num_test], y[i*num_test:(i+1)*num_test]
            train_X = X[:i*num_test] + X[(i+1)*num_test:]
            train_y = y[:i*num_test] + y[(i+1)*num_test:]
        train_data_loader = construct_loader(train_X, train_y, batch_size)

        model = make_model(**model_params)
        model.to(device)
        learning_rate = config["lr"]
        epoch = config["epoch"]

        train_model(model, train_data_loader, epoch, learning_rate)
        total_loss += test_model(model, test_X, test_y)
    print(f"END: Average loss of {total_loss/num_data} over {num_data} samples")
    tune.report(mean_loss=total_loss/num_data)


# -------------------------------------Set up params----------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
args = make_args()

batch_size = args.batch
learning_rate = args.lr
loss_fn = torch.nn.MSELoss(reduction='sum')
epoch = args.epoch
filename = './data/Mathieu/Mathieu_logH.csv'
if args.no_dbl_ring:
    filename = "./data/Mathieu/Mathieu_logH_noDblRing.csv"
if args.stratified:
    filename = "./data/Mathieu/Mathieu_logH_stratified.csv"
cv_fold = args.cv_fold
torch.manual_seed(args.seed)
logging.basicConfig(filename=f"{args.label}_{datetime.datetime.now().strftime('%b%d_%H%M')}.log",
                    level=logging.DEBUG)


# -----------------------------------------Load data----------------------------------------
X, y = load_data_from_df(filename, one_hot_formal_charge=True, atomic_charge=args.atomic_charge, geom=args.geom)
# Keep test set away from CV
X = X[:-args.withhold]
y = y[:-args.withhold]

# -------------------------------------Set parameters----------------------------------------
d_atom = X[0][0].shape[1]  # It depends on the used featurization.
model_params = {
    'd_atom': d_atom,
    'd_model': 1024,
    'N': 8,
    'h': 16,
    'N_dense': 1,
    'lambda_attention': 0.33, 
    'lambda_distance': 0.33,
    'leaky_relu_slope': 0.1, 
    'dense_output_nonlinearity': 'relu', 
    'distance_matrix_kernel': 'exp', 
    'dropout': args.dropout,
    'aggregation_type': 'mean'
}


# -------------------------------------Construct model--------------------------------------
search_space = {
    "lr": tune.grid_search([1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]),
    "epoch": tune.grid_search([100*x for x in range(1,9)]),
    }

ray.init(include_dashboard=False)
analysis = tune.run(
    cv,
    config=search_space,
    resources_per_trial={'gpu': 1, 'cpu': 10})

best_trial = analysis.get_best_trial(metric="mean_loss", mode="min")
print(f"Best config:\t{best_trial.config}")
print(f"Best loss:\t{best_trial.last_result['mean_loss']}")

# torch.save(model.state_dict(), "./train_all_scheduler.torch")
