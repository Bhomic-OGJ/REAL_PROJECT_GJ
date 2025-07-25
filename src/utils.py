# from rdkit import Chem
# from rdkit.Chem import Draw, Descriptors
# from PIL import Image
# from rdkit.Chem import QED

from rdkit.Chem import QED
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
import torch
from io import BytesIO
from torchviz import make_dot
import os

def smiles_to_image(smiles, size=(300, 300)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES string"
        img = Draw.MolToImage(mol, size=size)
        return img, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def compute_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES string"
        properties = {
            "QED": QED.qed(mol),
            "LogP": Descriptors.MolLogP(mol),
            "SAS": Descriptors.MolWt(mol) / 100
        }
        return properties, None
    except Exception as e:
        return None, f"Error: {str(e)}"
    

def visualize_molecular_graph(smiles, graph, size=(300, 300)):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES string"
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (atoms) with labels
        for idx, atom in enumerate(mol.GetAtoms()):
            G.add_node(idx, label=atom.GetSymbol())
        
        # Add edges (bonds) from edge_index
        edge_index = graph.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < dst:  # Avoid duplicate edges
                G.add_edge(src, dst)
        
        # Set up plot
        plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'label')
        
        # Draw graph
        nx.draw(
            G, pos,
            labels=labels,
            with_labels=True,
            node_color='lightblue',
            node_size=500,
            font_size=12,
            font_weight='bold',
            edge_color='gray'
        )
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return img, None
    except Exception as e:
        return None, f"Error: {str(e)}"
    

def visualize_gcn_architecture(model, graph, size=(300, 300)):
    try:
        # Perform a forward pass to create the computational graph
        model.eval()
        with torch.no_grad():
            output = model(graph)
        
        # Create a dot graph using torchviz
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        # Render to PNG
        dot.format = 'png'
        dot.render('gcn_architecture', cleanup=True)
        
        # Load as PIL Image
        img = Image.open('gcn_architecture.png')
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Clean up temporary file
        os.remove('gcn_architecture.png')
        
        return img, None
    except Exception as e:
        return None, f"Error: {str(e)}"