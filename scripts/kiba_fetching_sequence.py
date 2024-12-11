#%%
import pandas as pd 
import requests
from time import sleep
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Importing the kiba dataset
df_kiba_tot = pd.read_csv('/home/ali.dahaj/workspace/ribo_seq/data/Kiba_dataset.csv')

# Using the UniProt API to fetch protein sequences
def fetch_uniprot_sequences(uniprot_ids):
    base_url = "https://rest.uniprot.org/uniprotkb/"
    # Dictionary mapping UniProt IDs to their sequences (None if not found)
    sequences = {}
    # Number of ID to process in each batch
    batch_size = 100
    # Number of times to retry failed requests
    retry_attempts = 3
    # Time to wait between batches in seconds
    sleep_time = 1.0
    # Process IDs in batches
    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i + batch_size]
        
        for uniprot_id in batch:
            for attempt in range(retry_attempts):
                try:
                    response = requests.get(
                        f"{base_url}{uniprot_id}.fasta",
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        # Extract sequence from FASTA format
                        fasta_lines = response.text.split('\n')
                        sequence = ''.join(fasta_lines[1:]).strip()
                        sequences[uniprot_id] = sequence
                        break
                    elif response.status_code == 404:
                        sequences[uniprot_id] = None
                        print(f"Warning: {uniprot_id} not found")
                        break
                    else:
                        if attempt == retry_attempts - 1:
                            print(f"Error fetching {uniprot_id}: Status {response.status_code}")
                            sequences[uniprot_id] = None
                
                except requests.exceptions.RequestException as e:
                    if attempt == retry_attempts - 1:
                        print(f"Error fetching {uniprot_id}: {str(e)}")
                        sequences[uniprot_id] = None
                    
                sleep(0.1)  
        
        # Slep between batches to be nice to the API
        sleep(sleep_time)
        
        # Print progress
        print(f"Processed {min(i + batch_size, len(uniprot_ids))}/{len(uniprot_ids)} IDs")
    
    return sequences

# Fetch SMILES for a list of PubChem CIDs
def get_smiles(cids, max_workers=10):
    def fetch_single_smiles(cid):
        #Helper function to fetch single SMILSE
        try:
            response = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON",
                timeout=30
            )
            if response.status_code == 200:
                smiles = response.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
                return cid, smiles
            return cid, None
        except Exception as e:
            print(f"Failed to fetch CID {cid}: {str(e)}")
            return cid, None

    results = {}
    
    # Use ThreadPoolExecutor for parallel processing
    with tqdm(total=len(cids), desc="Fetching SMILES") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_cid = {executor.submit(fetch_single_smiles, cid): cid for cid in cids}
            
            # Process completed tasks
            for future in as_completed(future_to_cid):
                cid, smiles = future.result()
                if smiles:
                    results[cid] = smiles
                pbar.update(1)
                sleep(0.05)  
    
    return results

def generate_random_negatives(df, n_samples, protein_col='UniProt_ID', drug_col='pubchem_cid'):
    """Generate random negative pairs not in original data"""
    proteins = df[protein_col].unique()
    drugs = df[drug_col].unique()
    existing_pairs = set(map(tuple, df[[protein_col, drug_col]].values))
    
    negatives = []
    while len(negatives) < n_samples:
        # Generate batch of random pairs
        p = np.random.choice(proteins, size=min(n_samples * 2, 1000000))
        d = np.random.choice(drugs, size=min(n_samples * 2, 1000000))
        pairs = list(zip(p, d))
        
        # Filter valid pairs
        new_pairs = [pair for pair in pairs 
                    if pair not in existing_pairs 
                    and pair not in negatives]
        
        negatives.extend(new_pairs[:n_samples - len(negatives)])
    
    return pd.DataFrame(negatives, columns=[protein_col, drug_col])

#%%
# Remove rows with missing values
df_kiba_tot = df_kiba_tot.dropna()

# Remove rows with duplicate for the same protein AND drug
df_kiba_tot = df_kiba_tot.drop_duplicates(subset=['UniProt_ID', 'pubchem_cid'])

# Create negative samples by shuffling the Kiba dataset due to the imbalance
df_kiba_pos = df_kiba_tot[df_kiba_tot['kiba_score_estimated'] == True]
n_samples = 1000000
negatives = generate_random_negatives(df_kiba_pos, n_samples=n_samples)
# Create a smaller balanced dataset for testing
df_kiba_sampled = df_kiba_pos.sample(n=n_samples, random_state=42)

df_kiba = pd.concat([df_kiba_sampled, negatives])
df_kiba['kiba_score_estimated'] = df_kiba['kiba_score_estimated'].fillna(False)

print(df_kiba['pubchem_cid'].nunique())

#%%
column_name = 'UniProt_ID'
# Fetch protein sequences
uniprot_ids = df_kiba[column_name].unique().tolist()
sequences = fetch_uniprot_sequences(uniprot_ids)

# Save sequences dictionary to a CSV file
sequences_df = pd.DataFrame(list(sequences.items()), columns=['UniProt_ID', 'Sequence'])
sequences_df.to_csv('/home/ali.dahaj/workspace/ribo_seq/data/kiba_prot_seqs.csv', index=False)
# %%
column_name = 'pubchem_cid'
# Fetch SMILES for a subset of PubChem CIDs
pubchem_cids = df_kiba[column_name].unique().tolist()
pubchem_cids = [int(cid) for cid in pubchem_cids if pd.notna(cid)]
smiles = get_smiles(pubchem_cids)

# Save SMILES dictionary to a CSV file
smiles_df = pd.DataFrame(list(smiles.items()), columns=[column_name, 'SMILES'])
smiles_df.to_csv('/home/ali.dahaj/workspace/ribo_seq/data/kiba_smiles.csv', index=False)
# %%
# Merge sequences and smiles with the original dataframe
df_kiba['Protein_seqs'] = df_kiba['UniProt_ID'].map(sequences)
df_kiba['SMILES'] = df_kiba['pubchem_cid'].map(smiles)

# Create the final dataframe with the required columns
df_final = df_kiba[['Protein_seqs', 'SMILES', 'kiba_score_estimated']].rename(columns={'kiba_score_estimated': 'label'})

# Save the final dataframe to a CSV file
df_final.to_csv('/home/ali.dahaj/workspace/ribo_seq/data/kiba_final.csv', index=False)