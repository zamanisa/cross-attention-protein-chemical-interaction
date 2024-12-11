# Cross Attention Protein Chemical Interaction: CAPChI

A deep learning model using cross-attention mechanism to predict protein-chemical interactions.

## Overview

This model predicts protein-chemical binding interactions by:
- Processing protein sequences through ESM-2
- Encoding chemical SMILES through ChemBERTa
- Combining representations via cross-attention
- Fine-tuning for improved performance
- hyper parameter optimization using grid search method

## Installation

```bash
git clone https://github.com/yourusername/protein-chemical-interaction
cd protein-chemical-interaction
pip install -r requirements.txt
```

Required packages:
```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

## Usage

### Input Data Format
**Note:** The script `scripts/kiba_fetching_sequence.py` uses the Pubchem and Uniprot APIs to fetch sequence and SMILES format for the given Uniprot IDs and Pubchem cids. While it was successful for protein sequences, the Pubchem's API was returning 'PUGREST.ServerBusy' error. Hence I created a mock dataset.

Input data should be a CSV file with columns: UniProt_ID, PubChem_CID, label
See example format in `example_data/mock_kiba_data.csv`

```python
# Example input format
df_ids = pd.read_csv('example_data/mock_kiba_data.csv')
```
### Running the model
You can run the model by downloading CAPCHI.py from `scripts/` folder and modify input path:
```python
df = pd.read_csv('path/to/your/input.csv')
```
## Model Architecture

### Design Rationale

#### Encoders (ESM-2 and ChemBERTa)
Pre-trained encoders are essential because:
- Transform variable-length inputs into fixed-dimension embeddings
- Enable standardized vector representations for attention
- ESM-2 captures evolutionary patterns in protein sequences
- ChemBERTa learns chemical substructure representations
- Convert different data types (sequences/SMILES) into comparable formats
- Leverage domain knowledge from pre-training

#### Cross-Attention Mechanism
Cross-attention is crucial for interaction prediction because:
- Enables direct modeling between protein and chemical features
- Learns which protein regions interact with specific chemical substructures
- Maintains bidirectional information flow between modalities
- Weights feature importance based on interaction context
- Preserves spatial/sequential relationships in both inputs
  
**Note:** I have used the `outputs.last_hidden_state[0, 1:-1]` instead of `outputs.last_hidden_state[0, 1:-1]` for two main reasons: 1. It was computationally faster for the cross-attention layer and fine-tunning the models. 2. For all inputs the output has fixed length. However, it's better to use `outputs.last_hidden_state[0, 1:-1]` since the output is full sequence representations to enable attention between protein and chemical features and preserves position-specific information better. 
### Components
- ESM-2: Protein sequence encoder pre-trained on evolutionary data
- ChemBERTa: Chemical structure encoder trained on molecular data
- Cross-attention: Learns protein-chemical feature interactions
- Classification head: Predicts binding probability

### Architecture Benefits
- Fine-tuning capability for task-specific feature learning
- Bidirectional attention between protein and chemical features
- End-to-end training optimizes all components jointly
- Leverages pre-trained knowledge from both protein and chemical domains
- Can learn which parts of protein embeddings are relevant for specific chemical features and vice versa
- Can provide interpretability by examining attention weights

### Architecture Disadvantages
- More complex to implement and train
- Requires more computational resources
- Using the `[:,0,:]` indexing would only give you the [CLS] token embedding, which loses position-specific information needed for cross-attention.
