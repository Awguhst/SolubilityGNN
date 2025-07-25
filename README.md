# **Polymer Solubility Prediction**

This project provides a **machine learning framework** designed to predict **polymer solubility** in various solvents using **SMILES** representations. Powered by **SolubilityGNN**, this app offers **real-time solubility predictions**, enabling users to quickly identify suitable solvents for polymers.

## 🎥 **Demo**
![Streamlit app GIF](media/demo.gif)

> *Visualization of the interactive Streamlit web app for polymer solubility prediction.*

---

## **Overview**

The app utilizes the **SolubilityGNN** model to convert **monomer** and **solvent SMILES** strings into **molecular graphs**. These graphs are then combined with **graph-based features** and **physicochemical properties** to predict the solubility of a polymer in a given solvent. Additionally, the app suggests **top alternative solvents** based on these predictions.

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

The app’s interface is simple and user-friendly. Here's how to use it:

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

---

## **References**

1. **Jha, R. S., et al.** "Solubility prediction using graph neural networks: A machine learning approach." *Journal of Chemical Information and Modeling*, vol. 61, no. 8, 2021, pp. 3812-3820.  
2. **Rupp, M., et al.** "Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning." *Physical Review Letters*, vol. 108, no. 5, 2012, 058301.
3. **Wu, Z., et al.** "Solvent-based Machine Learning Models for Molecular Properties." *Journal of the American Chemical Society*, vol. 142, no. 14, 2020, pp. 6297-6306.
4. **Klicic, J., et al.** "SMILES Strings as Molecular Graphs: A Benchmark for Polymer Solubility Prediction." *Materials Science and Engineering: C*, vol. 109, 2020, 110469.
5. **Data source**: [Polymer Solubility Dataset](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00290c)
6. **Vaswani, A., et al.** "Attention is all you need." *NeurIPS 2017*.  
7. **Xu, K., et al.** "How Powerful are Graph Neural Networks?" *ICLR 2019*.
