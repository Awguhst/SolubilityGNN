# Polymer Solubility Prediction

This project offers a **machine learning framework** for predicting **polymer solubility** in various solvents based on **SMILES** representations. Using a **(SolubilityGNN)**, the app enables **real-time solubility predictions**, helping you quickly identify suitable solvents for polymers.

## 🎥 Demo
![Streamlit app GIF](media/demo.gif)

> *Visualization of the interactive Streamlit web app for polymer solubility prediction.*

---

## Overview

The app utilizes the **SolubilityGNN** model to convert monomer and solvent **SMILES** strings into molecular graphs. It then integrates **graph-based features** with **physicochemical properties** to predict the solubility of a polymer in a given solvent. The app also recommends **top alternative solvents** based on these predictions.

---

## Features

### Solubility Prediction
- **2D Visualization**: Displays **2D structures** of both the polymer and solvent from their SMILES strings for easy visualization.
- **Solubility Prediction**: Predicts the **probability** of polymer solubility in the solvent, shown as a **percentage**. Higher values indicate a stronger likelihood of solubility.
- **Solvent Classification**: Classifies solvents as **protic** or **aprotic**, reflecting their **hydrogen-bonding potential**, which impacts polymer-solvent interactions.
- **Polarity Estimation**: Uses **RDKit** to estimate the **polarity** of solvents, helping assess their dissolving power.
- **Top 5 Alternative Solvents**: Suggests **top 5 alternative solvents** ranked by predicted solubility, offering users more solvent options to explore.

---

### Streamlit App Interface

The app's interface is quite simple and easy to use.

1. **Enter Polymer SMILES**:  
   Input the **SMILES string** of the polymer (e.g., ``C#C``).

2. **Enter Solvent SMILES (Optional)**:  
   Optionally input the **SMILES string** of the solvent. If left blank, the app will use a **default solvent**.

3. **Click "Predict"**:  
   After entering the SMILES strings, click the **"Predict"** button. The app will display:
   - **Predicted Solubility** as a **probability score**.
   - **2D structures** of both the polymer and the solvent.
   - A list of the **top 5 alternative solvents**, including their `solubility predictions`, `solvent types`, and `polarities`.

---

## Model Architecture

The **SolubilityGNN** model processes both polymer and solvent as molecular graphs. The model’s key components include:

- **TransformerConv Layers**: Capture molecular interactions between atoms and bonds through bond and edge features.
- **Batch Normalization**: Stabilizes training and accelerates convergence.
- **GINConv Layer**: A **Graph Isomorphism Network** that extracts higher-level molecular patterns without relying on edge attributes.
- **Fully Connected Layers**: Combine graph features with molecular descriptors to predict solubility, utilizing **GELU activations** and **dropout** to prevent overfitting.
- **Prediction**: Outputs a single **probability score**, representing the likelihood of the polymer dissolving in the solvent.

---

## Model Performance

- **Validation Accuracy**: **82%** (5-fold cross-validation)
- **Validation AUC (Area Under the Curve)**: **0.88** (5-fold cross-validation)
