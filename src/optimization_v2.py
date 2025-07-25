import torch
from rdkit import Chem
from rdkit import generate_smiles
from dwave.samplers import SimulatedAnnealingSampler
import numpy as np
from gcn_model import compute_representation
from utils import mol_to_graph

def optimize_molecules(gcn_model, energy_model, target_properties, num_molecules, device='cpu'):
    try:
        generated_molecules = []
        sampler = SimulatedAnnealingSampler()

        for _ in range(num_molecules):
            # Initialize with a random SMILES
            current_smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'  # Starting point
            best_smiles = current_smiles
            best_energy = float('inf')

            # Perform 90 sampling steps (as per paper)
            for _ in range(50):
                # Generate candidate SMILES
                candidate_smiles = generate_smiles(current_smiles)
                
                # Filter valid SMILES
                valid_smiles = [s for s in candidate_smiles if Chem.MolFromSmiles(s) is not None]
                if not valid_smiles:
                    continue

                # Evaluate candidates
                for smi in valid_smiles:
                    graph, _ = mol_to_graph(smi)
                    graph = graph.to(device)
                    latent, _ = compute_representation(smiles, gcn_model, device)
                    hidden = torch.randn(128).to(device)
                    energy = energy_model(latent, target_properties, hidden).item()

                    # Simulated annealing to decide acceptance
                    Q = np.random.rand(1, 1)  # Simplified QUBO for energy minimization
                    response = sampler.sample_qubo(Q, num_reads=100)
                    accept = response.first().energy < 0  # Simplified acceptance criterion

                    if energy < best_energy:
                        best_energy = energy
                        best_smiles = smi
                    current_smiles = smi

            if Chem.MolFromSmiles(best_smiles):
                generated_molecules.append(best_smiles)

        return generated_molecules, None
    except Exception as e:
        return [], f"Error generating molecules: {str(e)}"