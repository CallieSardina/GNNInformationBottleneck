# training gnn models, we already shared trained ones at data/{dataset_name}/gnn

import argparse
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from models.GCN import GCN
import utils.data as data
from utils.edge_estimator import EDGE


def train(model, optimizer, train_loader, epoch, device, args):
    model.train()
    total_loss = 0

    for batch_idx, train_batch in tqdm(enumerate(train_loader), desc='Train Batch', total=len(train_loader)):
        optimizer.zero_grad()
        node_embeddings, graph_embedding, embeddings_per_layer, logits = model(train_batch.to(device))

        mi_XZ = [EDGE(embeddings_per_layer[0].clone().detach().numpy(), embeddings.clone().detach().numpy()) for embeddings in embeddings_per_layer]
        mi_ZY = [EDGE(train_batch.y.clone().detach().numpy(), embeddings.clone().detach().numpy()) for embeddings in embeddings_per_layer]
        
        with open(f'./MI_logs/{args.model}/mi_log_{args.num_layers}_layers.txt', 'a') as f:
            f.write(f"Epoch {epoch}, Batch {batch_idx}, "
                    f"MI_XZ: {', '.join(map(str, mi_XZ))}, "
                    f"MI_ZY: {', '.join(map(str, mi_ZY))}\n")

        loss = F.nll_loss(logits, train_batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * train_batch.num_graphs

    return total_loss / len(train_loader.dataset)
 

@torch.no_grad()
def eval(model, eval_loader, device):
    model.eval()
    total_loss = 0
    total_hits = 0

    for eval_batch in tqdm(eval_loader, desc='Eval Batch', total=len(eval_loader)):
        logits = model(eval_batch.to(device))[-1]
        loss = F.nll_loss(logits, eval_batch.y)
        total_loss += loss.item() * eval_batch.num_graphs
        pred = torch.argmax(logits, dim=-1)
        hits = (pred == eval_batch.y).sum()
        total_hits += hits

    return total_loss / len(eval_loader.dataset), total_hits / len(eval_loader.dataset)


def split_data(dataset, valid_ratio=0.1, test_ratio=0.1):
    valid_size = floor(len(dataset) * valid_ratio)
    test_size = floor(len(dataset) * test_ratio)
    train_size = len(dataset) - valid_size - test_size
    splits = torch.utils.data.random_split(dataset, lengths=[train_size, valid_size, test_size])

    return splits


def load_trained_gnn(dataset_name, device, args):
    dataset = data.load_dataset(dataset_name)
    model = GCN(
        num_features=dataset.num_features,
        num_classes=2,
        num_layers=args.num_layers,
        dim=20,
        dropout=0.0
    ).to(device)
    model.load_state_dict(torch.load(f'data/{dataset_name}/{args.model}_{args.num_layers}/model_best.pth', map_location=device))
    return model


@torch.no_grad()
def load_trained_prediction(dataset_name, device, args):
    prediction_file = f'data/{dataset_name}/{args.model}_{args.num_layers}/preds.pt'
    if os.path.exists(prediction_file):
        return torch.load(prediction_file, map_location=device)
    else:
        dataset = data.load_dataset(dataset_name)
        model = load_trained_gnn(dataset_name, device, args)
        model.eval()

        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        preds = []
        for eval_batch in tqdm(loader, desc='Eval Batch', total=len(loader)):
            logits = model(eval_batch.to(device))[-1]
            pred = torch.argmax(logits, dim=-1)
            preds.append(pred)
        preds = torch.cat(preds)
        torch.save(preds, prediction_file)
        return preds


@torch.no_grad()
def load_trained_embeddings_logits(dataset_name, device, args):
    node_embeddings_file = f'data/{dataset_name}/{args.model}n_{args.num_layers}/node_embeddings.pt'
    graph_embeddings_file = f'data/{dataset_name}/{args.model}_{args.num_layers}/graph_embeddings.pt'
    logits_file = f'data/{dataset_name}/{args.model}_{args.num_layers}/logits.pt'
    if os.path.exists(node_embeddings_file) and os.path.exists(graph_embeddings_file) and os.path.exists(logits_file):
        node_embeddings = torch.load(node_embeddings_file)
        for i, node_embedding in enumerate(node_embeddings):  # every graph has different size
            node_embeddings[i] = node_embeddings[i].to(device)
        graph_embeddings = torch.load(graph_embeddings_file, map_location=device)
        logits = torch.load(logits_file, map_location=device)
        return node_embeddings, graph_embeddings, logits
    else:
        dataset = data.load_dataset(dataset_name)
        model = load_trained_gnn(dataset_name, device, args)
        model.eval()
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        graph_embeddings, node_embeddings, logits = [], [], []
        for eval_batch in tqdm(loader, desc='Eval Batch', total=len(loader)):
            node_emb, graph_emb, logit = model(eval_batch.to(device))
            max_batch_number = max(eval_batch.batch)
            for i in range(max_batch_number + 1):
                idx = torch.where(eval_batch.batch == i)[0]
                node_embeddings.append(node_emb[idx])
            graph_embeddings.append(graph_emb)
            logits.append(logit)
        graph_embeddings = torch.cat(graph_embeddings)
        logits = torch.cat(logits)
        torch.save([node_embedding.cpu() for node_embedding in node_embeddings], node_embeddings_file)
        torch.save(graph_embeddings, graph_embeddings_file)
        torch.save(logits, logits_file)
        return node_embeddings, graph_embeddings, logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mutagenicity',
                        help="Dataset. Options are ['mutagenicity', 'aids', 'nci1', 'proteins']. Default is 'mutagenicity'. ")
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate. Default is 0.0. ')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size. Default is 128.')
    parser.add_argument('--num_layers', type=int,
                        help='Number of GCN layers. Default is 3.')
    parser.add_argument('--dim', type=int, default=20,
                        help='Number of GCN dimensions. Default is 20. ')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Random seed for training. Default is 0. ')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs. Default is 1000. ')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default is 0.001. ')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Index of cuda device to use. Default is 0. ')
    parser.add_argument('--train', type=int, default=0,
                        help='Train=1, Test=0.')
    parser.add_argument('--model', type=str, default='gcn',
                        help='Model architecture.')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Load and split the dataset.
    dataset = data.load_dataset(args.dataset)
    train_set, valid_set, test_set = split_data(dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # Logging.
    gnn_folder = f'data/{args.dataset}/{args.model}_{args.num_layers}/'
    if not os.path.exists(gnn_folder):
        os.makedirs(gnn_folder)
    log_file = gnn_folder + 'log.txt'
    with open(log_file, 'w') as f:
        pass

    # Initialize the model.
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    model = GCN(
        num_features=dataset.num_features,
        num_classes=2,
        num_layers=args.num_layers,
        dim=args.dim,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.train == 1:
        # Training.
        start_epoch = 1
        epoch_iterator = tqdm(range(start_epoch, start_epoch + args.epochs), desc='Epoch')
        best_valid = float('inf')
        best_epoch = 0
        for epoch in epoch_iterator:
            train_loss = train(model, optimizer, train_loader, epoch, device, args)
            valid_loss, valid_acc = eval(model, valid_loader, device)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_epoch = epoch
                print("Best epoch: ", best_epoch)
                torch.save(model.state_dict(), gnn_folder + f'model_checkpoint{epoch}.pth')
                torch.save(optimizer.state_dict(), gnn_folder + f'optimizer_checkpoint{epoch}.pth')
            with open(log_file, 'a') as f:
                print(f'Epoch = {epoch}:', file=f)
                print(f'Train Loss = {train_loss:.4e}', file=f)
                print(f'Valid Loss = {valid_loss:.4e}', file=f)
                print(f'Valid Accuracy = {valid_acc:.4f}', file=f)

    # Testing.
    model.load_state_dict(torch.load(gnn_folder + f'model_checkpoint{best_epoch}.pth', map_location=device))
    torch.save(model.state_dict(), gnn_folder + f'model_best.pth')
    train_acc = eval(model, train_loader, device)[1]
    valid_acc = eval(model, valid_loader, device)[1]
    test_acc = eval(model, test_loader, device)[1]
    with open(log_file, 'a') as f:
        print(file=f)
        print(f"Best Epoch = {best_epoch}:", file=f)
        print(f"Train Accuracy = {train_acc:.4f}", file=f)
        print(f"Valid Accuracy = {valid_acc:.4f}", file=f)
        print(f"Test Accuracy = {test_acc:.4f}", file=f)


if __name__ == '__main__':
    main()