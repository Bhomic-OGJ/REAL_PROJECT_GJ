import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, latent_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def mol_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES string"
        
        # Get atom features (e.g., atomic number, degree)
        num_features = 5  # Atomic number, degree, valence, hybridization, aromaticity
        x = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetExplicitValence(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic())
            ]
            x.append(features)
        x = torch.tensor(x, dtype=torch.float)
        
        # Get edge indices
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        from torch_geometric.data import Data
        return Data(x=x, edge_index=edge_index), None
    except Exception as e:
        return None, f"Error: {str(e)}"

def compute_representation(smiles, model, device='cpu'):
    graph, error = mol_to_graph(smiles)
    if error:
        return None, error
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        latent = model(graph)
    return latent.mean(dim=0).cpu().numpy(), None