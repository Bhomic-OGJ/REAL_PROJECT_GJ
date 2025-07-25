import streamlit as st
import pandas as pd
import torch
from PIL import Image
import os
from rdkit import Chem
from rdkit.Chem import Draw
from src.gcn_model import GCNModel, compute_representation
from src.energy_model import EnergyModel, train_energy_model
from src.optimization import optimize_molecules
from src.utils import mol_to_graph, visualize_gcn_architecture, visualize_ebm_architecture

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Streamlit app
st.title('Quantum Computing-Assisted Molecular Design')
st.write('This application demonstrates molecular design using a GCN and an Energy-Based Model with quantum-inspired optimization.')

# Sidebar for dataset and training
st.sidebar.header('Configuration')
dataset_path = st.sidebar.text_input('Dataset Path', 'data/zinc250k.csv')
train_button = st.sidebar.button('Train Models')
visualize_gcn_button = st.sidebar.button('Visualize GCN Architecture')
visualize_ebm_button = st.sidebar.button('Visualize EBM Architecture')

# Initialize session state
if 'gcn_model' not in st.session_state:
    st.session_state.gcn_model = None
if 'energy_model' not in st.session_state:
    st.session_state.energy_model = None
if 'gcn_image' not in st.session_state:
    st.session_state.gcn_image = None
if 'ebm_image' not in st.session_state:
    st.session_state.ebm_image = None

# Step 1: Load and preprocess dataset
st.header('Step 1: Load Dataset')
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path)
    st.write(f'Loaded dataset with {len(df)} molecules.')
    st.dataframe(df.head())
else:
    st.error('Dataset file not found. Please provide a valid path.')

# Step 2: Train models
if train_button:
    if os.path.exists(dataset_path):
        with st.spinner('Training GCN and Energy-Based Model...'):
            try:
                # Initialize and train GCN
                gcn_model = GCNModel(node_feature_dim=9, hidden_dim=64).to(device)
                st.session_state.gcn_model = gcn_model

                # Train Energy-Based Model
                energy_model, error = train_energy_model(dataset_path, gcn_model, device=device)
                if error:
                    st.error(error)
                else:
                    st.session_state.energy_model = energy_model
                    st.success('Training completed successfully.')
            except Exception as e:
                st.error(f'Training failed: {str(e)}')
    else:
        st.error('Please provide a valid dataset path.')

# Step 3: Visualize GCN architecture
if visualize_gcn_button and st.session_state.gcn_model:
    with st.spinner('Generating GCN architecture visualization...'):
        try:
            # Create a sample molecule
            sample_smiles = df['smiles'].iloc[0]
            sample_graph, _ = mol_to_graph(sample_smiles)
            sample_graph = sample_graph.to(device)

            image, error = visualize_gcn_architecture(st.session_state.gcn_model, sample_graph)
            if error:
                st.error(error)
            else:
                st.session_state.gcn_image = image
                st.image(image, caption='GCN Architecture', use_column_width=True)
        except Exception as e:
            st.error(f'Visualization failed: {str(e)}')

# Step 4: Visualize EBM architecture
if visualize_ebm_button and st.session_state.energy_model:
    with st.spinner('Generating EBM architecture visualization...'):
        try:
            # Create sample inputs
            sample_smiles = df['smiles'].iloc[0]
            latent, _ = compute_representation(sample_smiles, st.session_state.gcn_model, device)
            properties = torch.tensor([df['QED'].iloc[0], df['LogP'].iloc[0], df['SAS'].iloc[0]], dtype=torch.float32).to(device)
            hidden = torch.randn(128).to(device)  # Sample hidden variable

            image, error = visualize_ebm_architecture(st.session_state.energy_model, latent, properties, hidden)
            if error:
                st.error(error)
            else:
                st.session_state.ebm_image = image
                st.image(image, caption='Energy-Based Model Architecture', use_column_width=True)
        except Exception as e:
            st.error(f'Visualization failed: {str(e)}')

# Step 5: Generate molecules
st.header('Step 3: Generate Molecules')
target_qed = st.slider('Target QED', 0.0, 1.0, 0.9)
target_logp = st.slider('Target LogP', -5.0, 5.0, 2.0)
target_sas = st.slider('Target SAS', 0.0, 10.0, 3.0)
num_molecules = st.number_input('Number of Molecules to Generate', min_value=1, max_value=10, value=5)
generate_button = st.button('Generate Molecules')

if generate_button:
    if st.session_state.gcn_model and st.session_state.energy_model:
        with st.spinner('Generating molecules...'):
            try:
                target_properties = torch.tensor([[target_qed, target_logp, target_sas]], dtype=torch.float32).to(device)
                generated_molecules, error = optimize_molecules(
                    st.session_state.gcn_model,
                    st.session_state.energy_model,
                    target_properties,
                    num_molecules,
                    device=device
                )
                if error:
                    st.error(error)
                else:
                    st.subheader('Generated Molecules')
                    for i, smiles in enumerate(generated_molecules):
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            img = Draw.MolToImage(mol)
                            st.image(img, caption=f'Molecule {i+1}: {smiles}', use_column_width=True)
                        else:
                            st.write(f'Molecule {i+1}: Invalid SMILES - {smiles}')
            except Exception as e:
                st.error(f'Generation failed: {str(e)}')
    else:
        st.error('Please train the models first.')
