import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc(x))
        return x

def compute_representation(smiles, model, device='cpu'):
    try:
        from utils import mol_to_graph
        graph, _ = mol_to_graph(smiles)
        graph = graph.to(device)
        model.eval()
        with torch.no_grad():
            latent = model(graph)
        return latent, None
    except Exception as e:
        return None, f"Error computing representation: {str(e)}"
