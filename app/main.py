import streamlit as st
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
from torch_geometric.data import Data
from rdkit.Chem import Descriptors
from torch_geometric.utils import from_smiles
from torch_geometric.nn import global_max_pool, global_mean_pool
from solubility_gnn import SolubilityGNN  
from utils import process_smiles_pair

# Helper Functions
def classify_solvent_type(solvent_smiles):
    """Classify the solvent as Protic, Aprotic, or Intermediate based on functional groups."""
    
    # Convert SMILES string to RDKit molecule object
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
    
    if solvent_mol is None:
        return "Invalid SMILES"
    
    # Define functional groups
    protic_groups = [
        Chem.MolFromSmiles('O'),  # Hydroxyl group (-OH)
        Chem.MolFromSmiles('N'),  # Amine group (-NH, -NH2)
        Chem.MolFromSmiles('S'),  # Thiol group (-SH)
        Chem.MolFromSmiles('C(=O)O'),  # Carboxyl group (-COOH)
    ]
    
    aprotic_groups = [
        Chem.MolFromSmiles('C(O)C'),  # Ether group (-O-)
        Chem.MolFromSmiles('C=O'),    # Carbonyl group
        Chem.MolFromSmiles('C(C)C'),  # Alkyl groups (-CH3, -CH2-)
        Chem.MolFromSmiles('C#N'),    # Nitrile group (-CN)
        Chem.MolFromSmiles('CCl'),    # Chlorine attached to carbon
    ]
    
    intermediate_groups = [
        Chem.MolFromSmiles('C=O'),  # Carbonyl group (both protic and aprotic solvents can have this)
        Chem.MolFromSmiles('O'),    # Esters (weaker H-bonding than alcohols)
        Chem.MolFromSmiles('N=O'),  # Nitro groups
        Chem.MolFromSmiles('C-O'),  # Ethers (-O-) in intermediates
    ]
    
    # Check for protic groups (strong hydrogen bond donors)
    for group in protic_groups:
        if solvent_mol.HasSubstructMatch(group):
            return "Protic"
    
    # Check for aprotic groups (no hydrogen bond donors)
    for group in aprotic_groups:
        if solvent_mol.HasSubstructMatch(group):
            return "Aprotic"
    
    # Check for intermediate (ambiguous) groups
    for group in intermediate_groups:
        if solvent_mol.HasSubstructMatch(group):
            return "Intermediate (Ambiguous)"
    
    # If no match, return unknown
    return "Unknown"

def estimate_solvent_polarity_rdkit(solvent_smiles):
    """Estimate polarity using RDKit's MolLogP and partial charge distribution."""
    
    # Convert SMILES to molecule object
    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
    
    if solvent_mol is None:
        return "Invalid SMILES", None, None
    
    # Calculate LogP (higher LogP means more non-polar)
    logP = Descriptors.MolLogP(solvent_mol)
        
    # Heuristic to classify polarity
    if logP < 0:
        polarity = "High polarity (hydrophilic)"
    elif logP < 2:
        polarity = "Moderate polarity"
    else:
        polarity = "Low polarity (hydrophobic)"
    
    return polarity

# Custom CSS
st.markdown("""
    <style>
        /* Base background & text */
        html, body, [class*="css"] {
            background-color: #0e1117;
            color: #f1f1f1;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Headings */
        h1, h2, h3 {
            color: #61dafb;
            font-weight: 700;
            letter-spacing: 1px;
        }

        /* Buttons */
        .stButton>button {
            color: white;
            background: linear-gradient(135deg, #1f77b4, #3a86ff);
            padding: 0.6rem 1.5rem;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            font-size: 1.1rem;
            transition: background 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #3a86ff, #1f77b4);
            cursor: pointer;
        }

        /* Input field styling */
        .stTextInput>div>div>input {
            background-color: #1c1c1c;
            color: #f1f1f1;
            border: 1.5px solid #3a86ff;
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 1rem;
            font-weight: 500;
            transition: border-color 0.3s ease;
        }
        .stTextInput>div>div>input:focus {
            border-color: #61dafb;
            outline: none;
            box-shadow: 0 0 8px #61dafb;
        }

        /* Form container */
        .stForm {
            background-color: #1a1a1a;
            padding: 25px 30px;
            border-radius: 14px;
            box-shadow: 0 8px 24px rgba(97, 218, 251, 0.15);
        }

        /* Metric boxes */
        .stMetric {
            background-color: #222222;
            color: #f1f1f1;
            border-radius: 12px;
            padding: 16px 20px;
            font-size: 1.1rem;
            font-weight: 600;
            box-shadow: inset 0 0 5px rgba(97, 218, 251, 0.2);
        }

        /* Center molecule image */
        .css-1d391kg img {
            border-radius: 16px;
            box-shadow: 0 0 20px rgba(97, 218, 251, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# Model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SolubilityGNN(128, 5, 0.3).to(device)
model.load_state_dict(torch.load("../models/solubility_gnn.pt", map_location=device))
model.eval().to(device)

# Load training data
solvent_data = pd.read_csv('../data/solvent_smiles.csv')

# Streamlit interface configuration
st.set_page_config(page_title="Polymer Solubility Prediction", layout="wide", page_icon="🧬")

# Header Section
st.title("Polymer Solubility Prediction")
st.markdown("""
    Predict the solubility of a polymer in a solvent based on their SMILES strings. 
    Enter the SMILES strings of the monomer and a solvent, and get an immediate prediction of their solubility!
""")

# Sidebar Section: Inputs for SMILES
st.sidebar.header("SMILES Input")

polymer_smiles_input = st.sidebar.text_input("Polymer SMILES:", "C[SiH](O[SiH3])c1ccccc1", help="Enter monomer structure in SMILES format")
solvent_smiles_input = st.sidebar.text_input("Solvent SMILES (Optional):", "ClC(Cl)Cl", help="Enter solvent structure in SMILES format")

# Check if polymer SMILES input is empty and set the Submit button to disabled if true
if not polymer_smiles_input:
    st.sidebar.warning("❌ Please enter a polymer SMILES string before submitting.")
    submit = st.sidebar.button("Predict", disabled=True)  # Disable the button if input is empty
else:
    submit = st.sidebar.button("Predict")  # Enable the button if input is provided

# Prediction Logic
if submit:
    with st.spinner("🔬 Analyzing molecules..."):
        # Check if SMILES are valid
        polymer_mol = Chem.MolFromSmiles(polymer_smiles_input)
        solvent_mol = Chem.MolFromSmiles(solvent_smiles_input) if solvent_smiles_input else None

        if polymer_mol is None:
            st.error("❌ Invalid polymer SMILES string. Please enter a valid SMILES.")
        elif solvent_mol is None and solvent_smiles_input:
            st.error("❌ Invalid solvent SMILES string. Please enter a valid SMILES.")
        else:
            # Process the input SMILES strings into graph objects
            g1, g2 = process_smiles_pair(polymer_smiles_input, solvent_smiles_input) if solvent_mol else process_smiles_pair(polymer_smiles_input, solvent_smiles_input)

            # Prepare tensors and batch indices
            g1.x = g1.x.float().to(device)
            g2.x = g2.x.float().to(device)
            descriptor_tensor1 = g1.descriptors.to(device)
            descriptor_tensor2 = g2.descriptors.to(device)

            g1.edge_index = g1.edge_index.to(device)
            g1.edge_attr = g1.edge_attr.to(device)
            g2.edge_index = g2.edge_index.to(device)
            g2.edge_attr = g2.edge_attr.to(device)

            batch_index1 = torch.zeros(g1.x.size(0), dtype=torch.long).to(device)
            batch_index2 = torch.zeros(g2.x.size(0), dtype=torch.long).to(device)

            # Scenario 1: Both polymer and solvent provided
            if solvent_mol:
                with torch.no_grad():
                    output, _ = model(g1.x, g1.edge_index, batch_index1, descriptor_tensor1, g1.edge_attr, g2.x, g2.edge_index, batch_index2, descriptor_tensor2, g2.edge_attr)

                solvent_characteristic = torch.sigmoid(output).cpu().numpy()[0][0]

                st.subheader("📉 Solubility Prediction")
                st.metric(label="Probability of Polymer Being Solvable in This Solvent", value=f"{solvent_characteristic * 100:.2f}%")

                if solvent_characteristic > 0.5:
                    st.success("✅ The polymer is likely solvable in this solvent!")
                else:
                    st.warning("⚠️ The polymer is likely insoluble in this solvent!")

            # Create columns for Polymer and Solvent visualization side by side
            col1, col2, col3 = st.columns([1, 1, 1])

            # First column: Monomer 2D Structure
            with col1:
                st.subheader("Monomer 2D Structure")
                mol_img_polymer = Draw.MolToImage(polymer_mol, size=(500, 500))  # Adjust size to 300x300
                st.image(mol_img_polymer, use_container_width=True)

            # Second column: Solvent 2D Structure (if available)
            if solvent_mol:
                with col2:
                    st.subheader("Solvent 2D Structure")
                    mol_img_solvent = Draw.MolToImage(solvent_mol, size=(500, 500))  # Adjust size to 300x300
                    st.image(mol_img_solvent, use_container_width=True)

            # Recommend Top 5 Solvents
            st.subheader("Top 5 Alternative Solvents")
            top_solvents = []

            for _, row in solvent_data.iterrows():
                solvent_smiles = row['solvent_smiles']
                solvent_name = row['solvent']  # Get the trivial solvent name
                solvent_mol = Chem.MolFromSmiles(solvent_smiles)
                g2 = process_smiles_pair(polymer_smiles_input, solvent_smiles)[1]

                g2.x = g2.x.float().to(device)
                descriptor_tensor2 = g2.descriptors.to(device)
                g2.edge_index = g2.edge_index.to(device)
                g2.edge_attr = g2.edge_attr.to(device)

                batch_index2 = torch.zeros(g2.x.size(0), dtype=torch.long).to(device)

                with torch.no_grad():
                    output, _ = model(g1.x, g1.edge_index, batch_index1, descriptor_tensor1, g1.edge_attr, g2.x, g2.edge_index, batch_index2, descriptor_tensor2, g2.edge_attr)

                solubility = torch.sigmoid(output).cpu().numpy()[0][0]
                top_solvents.append((solvent_name, solvent_smiles, solubility))  # Store name, smiles, and solubility

            # Sort and Display Top 5 Solvents
            top_solvents = sorted(top_solvents, key=lambda x: x[2], reverse=True)[:5]
            
            # Create a column layout for better visual organization
            for i, (solvent_name, solvent_smiles, solubility) in enumerate(top_solvents):
                # Create columns for each solvent (Image, Name, SMILES, and Solubility)
                col1, col2, col3 = st.columns([1, 3, 1])

                # First column: Solvent Image
                with col1:
                    solvent_mol = Chem.MolFromSmiles(solvent_smiles)
                    mol_img = Draw.MolToImage(solvent_mol, size=(500, 500))
                    st.image(mol_img, caption=f"**Solvent {i+1}**", use_container_width=True)

                # Second column: Solvent Details (Name, SMILES, and Solubility)
                with col2:
                    st.markdown(f"### **{solvent_name}**")
                    st.markdown(f"**SMILES**: `{solvent_smiles}`")
                    st.markdown(f"**Predicted Solubility**: `{solubility * 100:.2f}%`")

                    # Add solvent properties
                    solvent_type = classify_solvent_type(solvent_smiles)
                    polarity = estimate_solvent_polarity_rdkit(solvent_smiles)

                    st.markdown(f"**Solvent Type**: {solvent_type}")
                    st.markdown(f"**Solvent Polarity**: {polarity}")

                # Third column: Empty or Optional (can be used for more info if needed)
                with col3:
                    pass  # Empty space for better layout control, or you can add more details if necessary

