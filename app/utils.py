import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles
from rdkit.Chem import Descriptors

def calculate_descriptors(mol):
    """
    Calculate molecular descriptors for a given molecule.

    Args:
        mol (RDKit Mol object): Molecule for which to calculate descriptors.

    Returns:
        torch.Tensor: Descriptors as a tensor.
    """
    if mol is None:
        return torch.zeros(1, 5, dtype=torch.float)

    descriptors = [
        Descriptors.MolWt(mol),       # Molecular weight
        Descriptors.TPSA(mol),        # Topological polar surface area
        Descriptors.MolLogP(mol),     # LogP (octanol-water partition coefficient)
        Descriptors.NumHDonors(mol),  # Number of hydrogen bond donors
        Descriptors.NumHAcceptors(mol) # Number of hydrogen bond acceptors
    ]
    
    return torch.tensor(descriptors, dtype=torch.float).view(1, -1)

def get_edge_features(mol):
    """
    Extract edge features for a given molecule in SMILES form.
    
    Args:
        mol (RDKit Mol object): Molecule for which to extract edge features.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Edge indices and edge attributes.
    """
    bond_features = []
    edge_index = []

    bond_type_to_idx = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3
    }

    num_bond_types = 4  # One-hot length for bond types

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # One-hot encode the bond type
        bond_type = bond_type_to_idx.get(bond.GetBondType(), -1)
        bond_type_one_hot = [0] * num_bond_types
        if 0 <= bond_type < num_bond_types:
            bond_type_one_hot[bond_type] = 1
        else:
            continue  # Skip unrecognized bond types

        # Additional boolean edge features
        extra_features = [
            float(bond.GetIsConjugated()),  # Is conjugated (1 or 0)
            float(bond.IsInRing())          # Is part of a ring (1 or 0)
        ]

        features = bond_type_one_hot + extra_features

        # Add bidirectional edges (i, j) and (j, i)
        edge_index += [[i, j], [j, i]]
        bond_features += [features, features]

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_types + 2), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).T  # Shape [2, num_edges]
        edge_attr = torch.tensor(bond_features, dtype=torch.float)  # Shape [num_edges, feature_dim]

    return edge_index, edge_attr

def process_smiles_pair(smile1, smile2):
    """
    Process a pair of SMILES strings and return graph objects for both molecules.
    
    Args:
        smile1 (str): SMILES string for the first molecule (polymer).
        smile2 (str): SMILES string for the second molecule (solvent).
        
    Returns:
        Tuple[Data, Data]: Processed graph objects for both molecules.
    """
    mol1 = Chem.MolFromSmiles(smile1)
    mol2 = Chem.MolFromSmiles(smile2)

    # Create graph objects from SMILES
    g1 = from_smiles(smile1)
    g2 = from_smiles(smile2)

    # Ensure node features are floats
    g1.x = g1.x.float()
    g2.x = g2.x.float()

    # Compute edge features
    g1.edge_index, g1.edge_attr = get_edge_features(mol1)
    g2.edge_index, g2.edge_attr = get_edge_features(mol2)

    # Compute molecular descriptors
    descriptor_tensor1 = calculate_descriptors(mol1)
    descriptor_tensor2 = calculate_descriptors(mol2)

    # Add descriptors to graph objects
    g1.descriptors = descriptor_tensor1
    g2.descriptors = descriptor_tensor2

    # Return the graph pair
    return g1, g2