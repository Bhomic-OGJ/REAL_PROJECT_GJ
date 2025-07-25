# QC Molecular Design

This project implements a simplified version of the molecular design framework described in the paper *"Molecular design with automated quantum computing-based deep learning and optimization"* by Akshay Ajagekar and Fengqi You (npj Computational Materials, 2023). It uses a Graph Convolutional Network (GCN) for molecular representation, an energy-based model for property prediction, and simulated quantum annealing for molecule generation.

## Project Structure
```
qc_molecular_design/
│
├── data/
│   └── sample_molecules.csv  # Sample dataset
├── src/
│   ├── gcn_model.py          # GCN implementation
│   ├── energy_model.py       # Energy-based model
│   ├── optimization.py       # Simulated annealing
│   └── utils.py             # Utility functions
├── app.py                   # Streamlit app
├── requirements.txt         # Dependencies
├── README.md                # Documentation
└── run.sh                   # Run script
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd qc_molecular_design
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ensure you have Python 3.8+ and CUDA (optional) for GPU support.

## Running the App
1. Start the Streamlit app:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
   Or directly:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`.

## Usage
1. **Step 1: Molecular Representation**
   - Enter a SMILES string (e.g., `CC(=O)OC1=CC=CC=C1C(=O)O` for aspirin).
   - Compute the GCN-based latent representation.
   - View the molecule visualization and latent vector.

2. **Step 2: Energy-Based Learning**
   - Train an energy-based model using the sample dataset.
   - View the molecular properties (QED, LogP, SAS) for the input molecule.

3. **Step 3: Molecule Generation**
   - Select a target property (QED, LogP, or SAS) and desired value.
   - Adjust tolerance and generate molecules.
   - View generated molecules with their properties.

## Notes
- The dataset is a small sample due to the unavailability of the paper’s full dataset. Extend it by adding more SMILES strings and properties to `data/sample_molecules.csv`.
- Quantum annealing is simulated using the `simanneal` library since access to quantum hardware (e.g., D-Wave) is not available.
- The GCN and energy-based model are simplified versions of the paper’s models, adapted for classical computing.
- For actual quantum computing integration, refer to the authors’ GitHub (https://github.com/PEESEgroup/qc-camd) or contact them for the dataset and QC code.

## Citation
Ajagekar, A., & You, F. (2023). Molecular design with automated quantum computing-based deep learning and optimization. *npj Computational Materials*, 9, 143. https://doi.org/10.1038/s41524-023-01099-0[](https://www.nature.com/articles/s41524-023-01099-0)

## License
MIT License



sudo apt install graphviz