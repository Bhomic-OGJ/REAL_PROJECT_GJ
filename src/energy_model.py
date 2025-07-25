import torch
import torch.nn as nn
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from src.gcn_model import mol_to_graph  # Import mol_to_grap

class EnergyModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(EnergyModel, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Energy score
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_energy_model(dataset_path, gcn_model, epochs=2, device='cpu'):
    try:
        df = pd.read_csv(dataset_path)
        model = EnergyModel(latent_dim=64, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for smiles in df['SMILES']:
                graph, error = mol_to_graph(smiles)
                if error:
                    continue
                graph = graph.to(device)
                gcn_model.eval()
                with torch.no_grad():
                    latent = gcn_model(graph).mean(dim=0)
                
                mol = Chem.MolFromSmiles(smiles)
                target = torch.tensor([QED.qed(mol)], dtype=torch.float).to(device)
                
                energy = model(latent)
                loss = criterion(energy, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(df):.4f}")
        return model, None
    except Exception as e:
        return None, f"Error: {str(e)}"