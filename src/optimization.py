import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import math
import pandas as pd
import os

class MoleculeOptimizer:
    def __init__(self, target_property, target_value, tolerance, sample_smiles):
        self.target_property = target_property
        self.target_value = target_value
        self.tolerance = tolerance
        self.sample_smiles = sample_smiles
        self.state = np.random.choice(sample_smiles)
        self.best_state = self.state
        self.best_energy = float('inf')
    
    def energy(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float('inf')
        if self.target_property == "QED":
            value = QED.qed(mol)
        elif self.target_property == "LogP":
            value = Descriptors.MolLogP(mol)
        elif self.target_property == "SAS":
            value = Descriptors.MolWt(mol) / 100
        else:
            return float('inf')
        return abs(value - self.target_value)
    
    def move(self):
        return np.random.choice(self.sample_smiles)
    
    def anneal(self, steps=1000, Tmax=100.0, Tmin=0.1):
        current_energy = self.energy(self.state)
        if current_energy < self.best_energy:
            self.best_energy = current_energy
            self.best_state = self.state
        
        for step in range(steps):
            temp = Tmax * (Tmin / Tmax) ** (step / steps)
            new_state = self.move()
            new_energy = self.energy(new_state)
            
            if new_energy < self.best_energy:
                self.best_state = new_state
                self.best_energy = new_energy
            
            if new_energy < current_energy or \
               np.random.rand() < math.exp((current_energy - new_energy) / temp):
                self.state = new_state
                current_energy = new_energy
        
        return self.best_state, self.best_energy

def optimize_molecules(target_property, target_value, tolerance=0.1):
    try:
        # Load SMILES from dataset
        # dataset_path = os.path.join("data", "sample_molecules.csv")
        
        dataset_path = os.path.join("data", "zinc250k", "250k_rndm_zinc_drugs_clean_3.csv")
        
        if not os.path.exists(dataset_path):
            return [], f"Error: Dataset file {dataset_path} not found"
        df = pd.read_csv(dataset_path)
        if 'SMILES' not in df.columns:
            return [], "Error: 'SMILES' column not found in dataset"
        sample_smiles = df['SMILES'].tolist()
        if not sample_smiles:
            return [], "Error: No SMILES strings found in dataset"
        
        optimizer = MoleculeOptimizer(target_property, target_value, tolerance, sample_smiles)
        best_state, best_energy = optimizer.anneal()
        if best_energy <= tolerance:
            return [(best_state, best_energy)], None
        return [], "No molecules found within tolerance"
    except Exception as e:
        return [], f"Error: {str(e)}"