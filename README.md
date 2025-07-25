# **Polymer Solubility Prediction**

This project provides a **machine learning framework** designed to predict **polymer solubility** in various solvents using **SMILES** representations. Powered by **SolubilityGNN**, this app offers **real-time solubility predictions**, enabling users to quickly identify suitable solvents for polymers.

## 🎥 **Demo**
![Streamlit app GIF](media/app.gif)

> *Visualization of the interactive Streamlit web app for polymer solubility prediction.*

---

## **Overview**

The app utilizes the **SolubilityGNN** model to convert **monomer** and **solvent SMILES** strings into **molecular graphs**. These graphs are then combined with **graph-based features** and **physicochemical properties** to predict the solubility of a polymer in a given solvent. Additionally, the app suggests **top alternative solvents** based on these predictions.

---

## Features

- **2D Visualization**: Displays **2D structures** of both the polymer and solvent from their SMILES strings for easy visualization.
- **Solubility Prediction**: Predicts the **probability** of polymer solubility in the solvent, shown as a **percentage**. Higher values indicate a stronger likelihood of solubility.
- **Solvent Classification**: Classifies solvents as **protic** or **aprotic**, reflecting their **hydrogen-bonding potential**, which impacts polymer-solvent interactions.
- **Polarity Estimation**: Uses **RDKit** to estimate the **polarity** of solvents, helping assess their dissolving power.
- **Top 5 Alternative Solvents**: Suggests **top 5 alternative solvents** ranked by predicted solubility, offering users more solvent options to explore.

---

## Streamlit App Interface

The app’s interface is simple and user-friendly. Here's how to use it:

- **Enter Polymer SMILES**:  
   Input the **SMILES string** of the polymer (e.g., ``C#C``).

- **Enter Solvent SMILES (Optional)**:  
   Optionally input the **SMILES string** of the solvent. If left blank, the app will use a **default solvent**.

- **Click "Predict"**:  
   After entering the SMILES strings, click the **"Predict"** button. The app will display:
   - **Predicted Solubility** as a **probability score**.
   - **2D structures** of both the polymer and the solvent.
   - A list of the **top 5 alternative solvents**, including their `solubility predictions`, `solvent types`, and `polarities`.

---

## Model Architecture

The **SolubilityGNN** model processes both polymer and solvent as molecular graphs. The model’s key components include:

- **TransformerConv Layers**: Capture molecular interactions between atoms and bonds through bond and edge features.
- **Batch Normalization**: Stabilizes training and accelerates convergence.
- **GINConv Layer**: A **Graph Isomorphism Network** that extracts higher-level molecular patterns without relying on edge features.
- **Fully Connected Layers**: Combine graph features with molecular descriptors to predict solubility, utilizing **GELU activations** and **dropout** to prevent overfitting.
- **Prediction**: Outputs a single **probability score**, representing the likelihood of the polymer dissolving in the solvent.

---

## Model Performance

- **Validation Accuracy**: **82%** (5-fold cross-validation)
- **Validation AUC (Area Under the Curve)**: **0.88** (5-fold cross-validation)

---

## **References**

- **Feinberg, E. N., et al.** (2018). PotentialNet for Molecular Property Prediction. *ACS Central Science*, 4(11), 1520–1530.  
   [https://doi.org/10.1021/acscentsci.8b00507](https://doi.org/10.1021/acscentsci.8b00507)

- **Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E.** (2017). Neural Message Passing for Quantum Chemistry. *International Conference on Machine Learning (ICML)*.  
   [https://doi.org/10.48550/arXiv.1704.01212](https://doi.org/10.48550/arXiv.1704.01212)

- **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.  
   [https://jmlr.org/papers/v12/pedregosa11a.html](https://jmlr.org/papers/v12/pedregosa11a.html)

- **RDKit: Open-source cheminformatics software.** (2006).  
   Available at: [http://www.rdkit.org](http://www.rdkit.org)

- **Stubbs, C. D., et al.** (2025). Predicting homopolymer and copolymer solubility through machine learning. *Dalton Transactions*.  
   [https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00290c](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00290c)

- **Vaswani, A., et al.** (2017). Attention is all you need. *NeurIPS 2017*.  
   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

- **Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y.** (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.  
   [https://doi.org/10.48550/arXiv.1710.10903](https://doi.org/10.48550/arXiv.1710.10903)

- **Xu, K., Hu, W., Leskovec, J., & Jegelka, S.** (2019). How Powerful are Graph Neural Networks? *Proceedings of the International Conference on Learning Representations (ICLR)*.  
   [https://doi.org/10.48550/arXiv.1810.00826](https://doi.org/10.48550/arXiv.1810.00826)
