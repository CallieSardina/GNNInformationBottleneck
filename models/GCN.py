
import torch
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
import torch.nn.functional as F

class GCN(torch.nn.Module):

    def __init__(self, num_features, num_classes=2, num_layers=None, dim=20, dropout=0.0):
        super(GCN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dim = dim
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # First GCN layer.
        self.convs.append(GCNConv(num_features, dim))
        self.bns.append(torch.nn.BatchNorm1d(dim))

        # Follow-up GCN layers.
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(dim, dim))
            self.bns.append(torch.nn.BatchNorm1d(dim))

        # Fully connected layer.
        self.fc = torch.nn.Linear(dim, num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, GCNConv):
                m.reset_parameters()
            elif isinstance(m, torch.nn.BatchNorm1d):
                m.reset_parameters()
            elif isinstance(m, torch.nn.Linear):
                m.reset_parameters()

    def forward(self, data, edge_weight=None):

        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        embeddings_per_layer = []

        # GCNs.
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout after every layer.
            # embeddings_per_layer.append(x)
            graph_embedding_per_layer = global_mean_pool(x, batch)
            embeddings_per_layer.append(graph_embedding_per_layer)

        # Pooling and FCs.
        node_embeddings = x
        graph_embedding = global_max_pool(node_embeddings, batch)
        out = self.fc(graph_embedding)
        logits = F.log_softmax(out, dim=-1)

        return node_embeddings, graph_embedding, embeddings_per_layer, logits