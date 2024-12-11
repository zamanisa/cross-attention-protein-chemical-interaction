import pandas as pd
import random

NUM_SAMPLES = 10000  # Change this single parameter to adjust dataset size

def generate_protein_seq(length=10):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return ''.join(random.choice(amino_acids) for _ in range(length))

def generate_smiles():
    atoms = ['C', 'N', 'O', 'S', 'P']
    bonds = ['', '=', '#']
    length = random.randint(3, 8)
    smiles = ''
    for i in range(length):
        smiles += random.choice(atoms)
        if i < length-1:
            smiles += random.choice(bonds)
    return smiles
max_len = 1000
min_len = 100
data = {
    'protein_sequence': [generate_protein_seq(random.randint(min_len, max_len)) for _ in range(NUM_SAMPLES)],
    'smiles': [generate_smiles() for _ in range(NUM_SAMPLES)],
    'label': [random.choice([True, False]) for _ in range(NUM_SAMPLES)]
}

df = pd.DataFrame(data)
df.to_csv('/home/ali.dahaj/kiba/data/mock_kiba_data.csv', index=False)
