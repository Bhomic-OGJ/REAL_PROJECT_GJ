import torch
import torch.nn as nn
import pandas as pd
from src.utils import mol_to_graph
from src.gcn_model import compute_representation
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np

class EnergyModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim, hidden_latent_dim):
        super(EnergyModel, self).__init__()
        self.fc1 = nn.Linear(latent_dim + 3 + hidden_latent_dim, hidden_dim)  # +3 for QED, LogP, SAS
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.hidden_latent_dim = hidden_latent_dim
    
    def forward(self, latent, properties, hidden):
        x = torch.cat([latent, properties, hidden], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        energy = self.fc3(x)
        return energy

def train_energy_model(dataset_path, gcn_model, device='cpu'):
    try:
        df = pd.read_csv(dataset_path)
        model = EnergyModel(latent_dim=64, hidden_dim=128, hidden_latent_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sampler = SimulatedAnnealingSampler()

        for epoch in range(100):
            total_loss = 0
            for _, row in df.iterrows():
                smiles = row['smiles']
                graph, _ = mol_to_graph(smiles)
                graph = graph.to(device)
                latent, _ = compute_representation(smiles, gcn_model, device)
                properties = torch.tensor([row['QED'], row['LogP'], row['SAS']], dtype=torch.float32).to(device)
                
                # Sample hidden variables (positive phase)
                hidden = torch.randn(128).to(device)
                
                # Compute positive phase energy
                energy_pos = model(latent, properties, hidden)
                
                # Negative phase: Sample hidden and properties using simulated annealing
                Q = np.random.rand(128 + 3, 128 + 3)  # Simplified QUBO matrix
                response = sampler.sample_qubo(Q, num_reads=1)
                samples = np.array([list(sample.values()) for sample in response])
                hidden_neg = torch.tensor(samples[0][:128], dtype=torch.float32).to(device)
                properties_neg = torch.tensor(samples[0][128:], dtype=torch.float32).to(device)
                
                # Compute negative phase energy
                energy_neg = model(latent, properties_neg, hidden_neg)
                
                # Contrastive Divergence-like loss
                loss = energy_pos - energy_neg
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss / len(df)}')
        
        return model, None
    except Exception as e:
        return None, f"Error training energy model: {str(e)}"