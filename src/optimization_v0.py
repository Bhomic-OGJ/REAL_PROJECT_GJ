from simanneal import Annealer
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import numpy as np

class MoleculeOptimizer(Annealer):
    def __init__(self, target_property, target_value, tolerance, sample_smiles):
        self.target_property = target_property
        self.target_value = target_value
        self.tolerance = tolerance
        self.sample_smiles = sample_smiles
        # self.state = np.random.choice(sample_smiles)
        # super(MoleculeOptimizer, self).__init__()
        initial_state = np.random.choice(sample_smiles)
        super(MoleculeOptimizer, self).__init__(initial_state=initial_state)

    def move(self):
        self.state = np.random.choice(self.sample_smiles)
    
    def energy(self):
        mol = Chem.MolFromSmiles(self.state)
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

def optimize_molecules(target_property, target_value, tolerance=0.1):
    try:
        sample_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "CCN(CC)C(=O)[C@H]1CN(C)[C@@H]2Cc3c[nH]c4cccc(c34)[C@@H]1C2"
        ]

        print("HEHE")
        
        optimizer = MoleculeOptimizer(target_property, target_value, tolerance, sample_smiles)
        optimizer.steps = 1000
        optimizer.Tmax = 100.0
        optimizer.Tmin = 0.1
        
        print("optimizer", optimizer)

        best_state, best_energy = optimizer.anneal()
        if best_energy <= tolerance:
            return [(best_state, best_energy)], None
        return [], "No molecules found within tolerance"
    except Exception as e:
        return [], f"Error: {str(e)}"