import sys
print(sys.executable)
import sys
sys.path.append('./packages/gcn_interpretation/gnn-model-explainer')
sys.path.append('./packages/ldbExtraction')
from explain_graphs import *
import dnn_invariant.extract_rules as extract
import torch
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from edge_estimator import EDGE
import os

# configure libraries
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# torch.use_deterministic_algorithms(True)

# # configure GPU
# gpu_id = 0
# torch.cuda.set_device(gpu_id)
# torch.cuda.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# device = 'cuda:0'

# gpu_id = 0
# device = f'cuda:{gpu_id}'

device = torch.device("cuda")
print(f"Using device: {device}")

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())


# Ensure the chosen device is available
assert torch.cuda.is_available(), "CUDA is not available!"

# Set the device
# torch.cuda.set_device(gpu_id)  # Not always necessary if using device directly

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)  # If using multi-GPU

print(f"Using device: {device}")

print(torch.__version__)
print(torch.version.cuda)  # Should not be None
print(torch.backends.cudnn.version())  # Should not be None



# get arguments
args = get_mutagenicity_args()
args.num_epochs = 10

# set constants
train = True

# load ckpt containing model and dataset
ckpt = torch.load("./ckpt/Mutagenicity/Mutagenicity_base_h20_o20.pth.tar", weights_only=False)

cg_dict = ckpt["cg"]

# extract model from ckpt
input_dim = cg_dict["feat"].shape[2]
num_classes = cg_dict["pred"].shape[2]

model = get_mutagenicity_model(input_dim, num_classes, device, args).cuda()
model.load_state_dict(ckpt["model_state"])

# extract dataset from ckpt
adj, feat, label, pred, num_nodes = get_dataset_from_ckpt(ckpt)

print("Feature shape:", feat.shape)
print("Adjacency shape:", adj.shape)
print("Num nodes:", num_nodes)
print("Label shape: ", label.shape)

size = 3035
train_data = (adj[:size], feat[:size], label[:size], num_nodes[:size])
val_data = (adj[size - 100:], feat[size - 100:], label[size - 100:], num_nodes[size - 100:])

# get graphs to explain
graph_indices = []
preds = np.argmax(cg_dict['pred'][0,:,:], axis=1) 
for i, l in enumerate(preds):
    # explain predictions for class 0 (mutagenic) for Mutagenicity
    # if l == 0:
    graph_indices.append(i)

# extract rules for each graph
rule_dict = extract.extract_rules('Mutagenicity', train_data, val_data, args, model.state_dict(), graph_indices=None, pool_size=50)

explainer = ExplainModule(
    model = model, 
    num_nodes = adj.shape[1],
    emb_dims = model.embedding_dim * model.num_layers * 2, 
    device = device,
    args = args
)

# Load explainer model from file
state_dict = torch.load('./ckpt/Mutagenicity/RCExplainer/rcexplainer.pth.tar')

exp_state_dict = explainer.state_dict()
for name, param in state_dict.items():
    if name in exp_state_dict and not ("model" in name):
        exp_state_dict[name].copy_(param)
explainer.load_state_dict(state_dict)

# train
if train:
    explainer, embs, embs_dict = train_explainer(explainer, model, rule_dict, adj, feat, label, pred, num_nodes, args, graph_indices)

# mi_XZ_layers = {}  # MI between input and each layer
# mi_ZY_layers = {}  # MI between each layer and output labels

# for i in embs_dict:
#     embs_dict[i] = torch.cat(embs_dict[i], dim=0)

# true_Y_np = label.cpu().detach().numpy().reshape(-1, 1)   # Ensure correct shape

# print("Y shape: ", true_Y_np.shape)
# # print("Y:", true_Y_np)
# # print(true_Y_np[0])

# num_layers = len(embs_dict)

# for i, layer_embs in embs_dict.items():
#     layer_embs_np = np.array(layer_embs)  # Convert embeddings to NumPy
#     layer_embs_np = layer_embs_np.mean(axis=1)  # Mean over nodes → (3000, 20)


#     print(f"Emb layer {i} shape: ", layer_embs_np.shape)
    
#     layer_0_embs_np = np.array(embs_dict[0]).mean(axis=1) 

#     mi_XZ = [EDGE(layer_0_embs_np, layer_embs_np)]
#     mi_ZY = [EDGE(true_Y_np, layer_embs_np)]
    
#     # Store the averaged MI values for this layer
#     mi_XZ_layers[i] = np.mean(mi_XZ)
#     mi_ZY_layers[i] = np.mean(mi_ZY)

#     print(f"Layer {i}: Avg MI_XZ = {mi_XZ_layers[i]:.4f}, Avg MI_ZY = {mi_ZY_layers[i]:.4f}")

# log_filename = f'./MI_logs/mi_log_layers.txt'
# with open(log_filename, 'a') as f:
#     for i in mi_XZ_layers:
#         f.write(f"Layer {i}: Avg MI_XZ: {mi_XZ_layers[i]:.4f}, Avg MI_ZY: {mi_ZY_layers[i]:.4f}\n")


# explain and model
explanations = evaluate_explainer(explainer, model, rule_dict, adj, feat, label, pred, num_nodes, args, graph_indices)

visualization_index = 2
feat_tmp = feat[visualization_index]
adj_tmp = adj[visualization_index]
explanation = explanations[visualization_index]

nz_idx, nz = np.nonzero(feat_tmp.numpy())
node_labels = {idx: nz[idx] for idx in nz_idx}

node_lut = {
    0: 'C',
    1: 'O',
    2: 'Cl', 
    3: 'H',
    4: 'N',
    5: 'F',
    6: 'Br', 
    7: 'S', 
    8: 'P', 
    9: 'I',
    10: 'Na', 
    11: 'K', 
    12: 'Li', 
    13: 'Ca'
}

node_colors = [node_labels[idx] for idx in nz_idx]

G = nx.Graph()

for idx in nz_idx:
    G.add_node(idx, atom=node_lut[node_labels[idx]], color=node_colors[idx])

edge_labels = {}

x, y = np.nonzero(adj_tmp.numpy())
for i in range(len(x)):
    edge_labels[(x[i], y[i])] = explanation[x[i], y[i]]

for a, b in edge_labels.keys():
    color = "r" if edge_labels[(a,b)] > 0.1 else "black"
    G.add_edge(a, b, color=color, wdith='1.0', weight="{:1.2f}".format(edge_labels[(a, b)]))

pos = nx.spring_layout(G, weight=1.0)
plt.figure()
node_labels = nx.get_node_attributes(G, "atom")

edges = G.edges()
edge_colors = [G[u][v]['color'] for u, v in edges]

nodes = G.nodes()
node_colors = nx.get_node_attributes(G, 'color')
node_colors = [node_colors[i] for i in node_colors.keys()] 
nx.draw(G, pos, labels=node_labels, node_color=node_colors, edge_color=edge_colors)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# nx.draw_networkx_edge_labels(G, pos)
plt.title("Explanations on sample " + str(visualization_index) + " of Mutagenicity dataset")
plt.savefig("mol.png")





