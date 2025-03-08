from explainer_main import main
from argparse import ArgumentParser
import json
import os
import gdown

#os.environ["WANDB_MODE"] = "offline"  # Use wandb logging in offline mode

# Create empty saved_models directory
saved_models_dir = './saved_models_new'

SEEDS = [0, 1, 3, 5, 8, 10, 15, 42, 69, 101]  # All seeds during training and evaluating
SPARSITIES = [0.8, 1.0]  # Different train/tests splits: 0.8 represents 80/20 split; and 1.0 represents 100/100 split

# Create namespace with used hyperparameters
parser = ArgumentParser()
prog_args = parser.parse_args('')

with open('commandline_args.txt', 'r') as f:
    prog_args.__dict__ = json.load(f)

TRAIN = True 

def train(args):
    if TRAIN is False:
        return
    
    for sparsity in SPARSITIES:
        args.train_data_sparsity = sparsity  # Set train/test split for all seeds
        for seed in SEEDS:
            args.seed = seed  # Set seed for training iteration
            main(args)

# Train RCExplainer from scratch
prog_args.num_epochs = 600
prog_args.bmname = "Mutagenicity"  # Dataset used for training
prog_args.explainer_method = "rcexplainer"  # Explainer model to train
prog_args.prefix = "rcexp_mutag"  # Used for storing models and logging training

train(prog_args)

from get_results import get_all_results
results_dir = "./results"

# Copy all models to saved_models directory for evaluating
import shutil
import os

if TRAIN is True:
    # Ensure the target directory exists
    os.makedirs('saved_models', exist_ok=True)
    
    # Copy files from 'ckpt/Mutagenicity' to 'saved_models'
    source_dir = 'ckpt/Mutagenicity'
    target_dir = 'saved_models'
    
    # Recursively copy all files from source to target
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        t = os.path.join(target_dir, item)
        
        if os.path.isdir(s):
            shutil.copytree(s, t)  # Recursively copy directories
        else:
            shutil.copy2(s, t)  # Copy files

print(get_all_results(results_dir))

from plot_graphs import plot_main

# Weird Mac OSX stuff
# !rm -r RESULTS/.DS_STORE

plot_main(results_dir, 'fidelity', threshold=0, figsize=(16,6), title='fid')
plot_main(results_dir, 'noise', figsize=(16,6), title='noise')
