import streamlit as st
import pandas as pd
import torch
from src.gcn_model import GCN, compute_representation, mol_to_graph
from src.energy_model import EnergyModel, train_energy_model
from src.optimization import optimize_molecules
from src.utils import smiles_to_image, compute_properties, visualize_molecular_graph, visualize_gcn_architecture
import streamlit.components.v1 as components

# Custom CSS for Tailwind styling
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    .main-container { @apply p-6 bg-gray-50 min-h-screen; }
    .card { @apply bg-white p-4 rounded-lg shadow-md mb-4; }
    .title { @apply text-2xl font-bold mb-4; }
    .subtitle { @apply text-xl font-semibold mb-2; }
    .button { @apply bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'molecules' not in st.session_state:
    st.session_state.molecules = []
if 'descriptors' not in st.session_state:
    st.session_state.descriptors = pd.DataFrame()
if 'gcn_model' not in st.session_state:
    st.session_state.gcn_model = GCN(num_features=5, hidden_dim=128, latent_dim=64)
if 'energy_model' not in st.session_state:
    st.session_state.energy_model = None

# Sidebar navigation
st.sidebar.title("QC Molecular Design")
page = st.sidebar.radio("Select Step", [
    "Step 1: Molecular Representation",
    # "Step 2: Energy-Based Learning",
    # "Step 3: Molecule Generation"
])

# Main app
st.markdown('<div class="main-container">', unsafe_allow_html=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.session_state.gcn_model.to(device)

# if page == "Step 1: Molecular Representation":
#     st.markdown('<div class="card"><h1 class="title">Step 1: Compute Molecular Representation</h1>', unsafe_allow_html=True)
#     st.write("Enter a SMILES string to compute its GCN-based molecular representation.")
#     smiles = st.text_input("SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
    
#     if st.button("Compute Representation", key="step1"):
#         latent, error = compute_representation(smiles, st.session_state.gcn_model, device)
#         img, img_error = smiles_to_image(smiles)
#         if error or img_error:
#             st.error(error or img_error)
#         else:
#             st.image(img, caption="Molecule Visualization", use_column_width=True)
#             st.write(f"**Latent Representation**: {latent[:10]}... (length: {len(latent)})")
#             st.session_state.molecules = [(smiles, img)]
#     st.markdown('</div>', unsafe_allow_html=True)


# if page == "Step 1: Molecular Representation":
#     st.markdown('<div class="card"><h1 class="title">Step 1: Compute Molecular Representation</h1>', unsafe_allow_html=True)
#     st.write("Enter a SMILES string to compute its GCN-based molecular representation and visualize the molecular graph.")
#     smiles = st.text_input("SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
    
#     if st.button("Compute Representation", key="step1"):
#         # Compute latent representation
#         latent, error = compute_representation(smiles, st.session_state.gcn_model, device)
#         # Get molecular image
#         img, img_error = smiles_to_image(smiles)
#         # Get molecular graph
#         graph, graph_error = mol_to_graph(smiles)
#         graph_img, graph_img_error = None, None
#         if graph:
#             graph_img, graph_img_error = visualize_molecular_graph(smiles, graph, size=(300, 300))
        
#         if error or img_error or graph_error or graph_img_error:
#             st.error(error or img_error or graph_error or graph_img_error)
#         else:
#             st.image(img, caption="Molecule Visualization", use_column_width=True)
#             st.image(graph_img, caption="Molecular Graph (GNN Input)", use_column_width=True)
#             st.write(f"**Latent Representation**: {latent[:10]}... (length: {len(latent)})")
#             st.session_state.molecules = [(smiles, img)]
#     st.markdown('</div>', unsafe_allow_html=True)


if page == "Step 1: Molecular Representation":
    st.markdown('<div class="card"><h1 class="title">Step 1: Compute Molecular Representation</h1>', unsafe_allow_html=True)
    st.write("Enter a SMILES string to compute its GCN-based molecular representation and visualize the molecular graph.")
    smiles = st.text_input("SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
    
    if st.button("Compute Representation", key="step1"):
        # Compute latent representation
        latent, error = compute_representation(smiles, st.session_state.gcn_model, device)
        # Get molecular image
        img, img_error = smiles_to_image(smiles)
        # Get molecular graph
        graph, graph_error = mol_to_graph(smiles)
        graph_img, graph_img_error = None, None
        if graph:
            graph_img, graph_img_error = visualize_molecular_graph(smiles, graph, size=(300, 300))
        
        if error or img_error or graph_error or graph_img_error:
            st.error(error or img_error or graph_error or graph_img_error)
        else:
            st.image(img, caption="Molecule Visualization", use_column_width=True)
            st.image(graph_img, caption="Molecular Graph (GNN Input)", use_column_width=True)
            st.write(f"**Latent Representation**: {latent[:10]}... (length: {len(latent)})")
            st.session_state.molecules = [(smiles, img)]
    st.markdown('</div>', unsafe_allow_html=True)


# the gcnn representation seems not usef
# if page == "Step 1: Molecular Representation":
#     st.markdown('<div class="card"><h1 class="title">Step 1: Compute Molecular Representation</h1>', unsafe_allow_html=True)
#     st.write("Enter a SMILES string to compute its GCN-based molecular representation, visualize the molecular graph, and the GCNN architecture.")
#     smiles = st.text_input("SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
    
#     if st.button("Compute Representation", key="step1"):
#         # Compute latent representation
#         latent, error = compute_representation(smiles, st.session_state.gcn_model, device)
#         # Get molecular image
#         img, img_error = smiles_to_image(smiles)
#         # Get molecular graph
#         graph, graph_error = mol_to_graph(smiles)
#         graph_img, graph_img_error = None, None
#         gcn_img, gcn_img_error = None, None
#         if graph:
#             graph = graph.to(device)
#             graph_img, graph_img_error = visualize_molecular_graph(smiles, graph, size=(300, 300))
#             # Get GCN architecture
#             gcn_img, gcn_img_error = visualize_gcn_architecture(st.session_state.gcn_model, graph, size=(300, 300))
        
#         if error or img_error or graph_error or graph_img_error or gcn_img_error:
#             st.error(error or img_error or graph_error or graph_img_error or gcn_img_error)
#         else:
#             st.image(img, caption="Molecule Visualization", use_column_width=True)
#             st.image(graph_img, caption="Molecular Graph (GNN Input)", use_column_width=True)
#             st.image(gcn_img, caption="GCNN Architecture", use_column_width=True)
#             st.write(f"**Latent Representation**: {latent[:10]}... (length: {len(latent)})")
#             st.session_state.molecules = [(smiles, img)]
#     st.markdown('</div>', unsafe_allow_html=True)

elif page == "Step 2: Energy-Based Learning":
    st.markdown('<div class="card"><h1 class="title">Step 2: Energy-Based Learning</h1>', unsafe_allow_html=True)
    st.write("Train an energy-based model using a dataset (simulated QC assistance).")
    
    if st.session_state.molecules:
        smiles, img = st.session_state.molecules[0]
        st.image(img, caption=f"Molecule: {smiles}", use_column_width=True)
        dataset_path = "./data/zinc250k/250k_rndm_zinc_drugs_clean_3.csv"
        
        if st.button("Train Energy Model", key="step2"):
            model, error = train_energy_model(dataset_path, st.session_state.gcn_model, device=device)
            print("model: ",model, "error: ", error)
            
            if error:
                st.error(error)
            else:
                st.session_state.energy_model = model
                properties, prop_error = compute_properties(smiles)

                print("properties: ",properties, " | prop_error: ",prop_error)

                if prop_error:
                    st.error(prop_error)
                else:
                    df = pd.DataFrame([properties])
                    st.session_state.descriptors = df
                    st.subheader("Molecular Properties")
                    st.dataframe(df.style.format("{:.2f}"))
    else:
        st.warning("Please compute a molecular representation in Step 1 first.")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Step 3: Molecule Generation":
    st.markdown('<div class="card"><h1 class="title">Step 3: Molecule Generation</h1>', unsafe_allow_html=True)
    st.write("Generate molecules with target properties using simulated QC optimization.")
    
    target_property = st.selectbox("Target Property", ["QED", "LogP", "SAS"])
    target_value = st.slider("Target Value", 0.0, 1.0 if target_property == "QED" else 5.0, 0.5)
    tolerance = st.slider("Tolerance", 0.01, 0.5, 0.1)
    


    if st.button("Generate Molecules", key="step3"):
        generated, error = optimize_molecules(target_property, target_value, tolerance)

        print("generated: ", generated, " | error: ", error)

        if error:
            st.error(error)
        else:
            if generated:
                st.subheader("Generated Molecules")
                for smi, energy in generated:
                    img, img_error = smiles_to_image(smi)
                    if img_error:
                        st.error(img_error)
                    else:
                        properties, prop_error = compute_properties(smi)
                        if prop_error:
                            st.error(prop_error)
                        else:
                            st.image(img, caption=f"SMILES: {smi}, {target_property}: {properties[target_property]:.2f}")
            else:
                st.warning(f"No molecules found within tolerance {tolerance} for {target_property} = {target_value}.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)