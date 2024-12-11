#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM

#Input dataframe
df= pd.read_csv('/home/ali.dahaj/workspace/ribo_seq/data/mock_kiba_data.csv')

class FineTunedModel(nn.Module):
    def __init__(self, protein_dim=1280, chemical_dim=600, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Load pre-trained models
        self.esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
        
        # Cross-attention components
        self.protein_projection = nn.Linear(protein_dim, hidden_dim)
        self.chemical_projection = nn.Linear(chemical_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, protein_seq, chemical_smiles):
        # Get embeddings directly from models
        protein_output = self.esm(**protein_seq).last_hidden_state[:,0,:]
        chemical_output = self.chemberta(**chemical_smiles, output_hidden_states=True)[0][:,0,:]
        
        # Rest of the cross-attention logic
        protein_hidden = self.protein_projection(protein_output)
        chemical_hidden = self.chemical_projection(chemical_output)
        
        protein_hidden = protein_hidden.unsqueeze(0)
        chemical_hidden = chemical_hidden.unsqueeze(0)
        
        attn_output, _ = self.cross_attention(protein_hidden, chemical_hidden, chemical_hidden)
        combined = torch.cat([attn_output.squeeze(0), protein_hidden.squeeze(0)], dim=1)
        
        return self.classifier(combined)

# Update data processing to pass raw sequences
def process_raw_data(df):
    tokenizer_protein = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    tokenizer_chemical = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    
    protein_inputs = tokenizer_protein(df['protein_sequence'].tolist(), return_tensors="pt", padding=True)
    chemical_inputs = tokenizer_chemical(df['smiles'].tolist(), return_tensors="pt", padding=True)
    labels = torch.tensor(df['label'].values, dtype=torch.float).view(-1, 1)
    
    return protein_inputs, chemical_inputs, labels

# The dataset class to handle the data
class InteractionDataset(Dataset):
    def __init__(self, protein_inputs, chemical_inputs, labels):
        self.protein_inputs = protein_inputs
        self.chemical_inputs = chemical_inputs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        protein = {k: v[idx] for k, v in self.protein_inputs.items()}
        chemical = {k: v[idx] for k, v in self.chemical_inputs.items()}
        return protein, chemical, self.labels[idx]


class EarlyStopping:
    def __init__(self, patience=100, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Update the training loop to handle the new data format
def train_with_validation(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    early_stopping = EarlyStopping(patience=100)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for protein_inputs, chemical_inputs, labels in train_loader:
            # Move all inputs to device
            protein_inputs = {k: v.to(device) for k, v in protein_inputs.items()}
            chemical_inputs = {k: v.to(device) for k, v in chemical_inputs.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(protein_inputs, chemical_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for protein_inputs, chemical_inputs, labels in val_loader:
                protein_inputs = {k: v.to(device) for k, v in protein_inputs.items()}
                chemical_inputs = {k: v.to(device) for k, v in chemical_inputs.items()}
                labels = labels.to(device)
                outputs = model(protein_inputs, chemical_inputs)
                val_loss += criterion(outputs, labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    return model

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for protein_inputs, chemical_inputs, labels in val_loader:
            protein_inputs = {k: v.to(device) for k, v in protein_inputs.items()}
            chemical_inputs = {k: v.to(device) for k, v in chemical_inputs.items()}
            labels = labels.to(device)
            outputs = model(protein_inputs, chemical_inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def grid_search(train_loader, val_loader, device):
    param_grid = {
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'hidden_dim': [256, 512, 1024],
        'num_heads': [4, 8],
        'dropout': [0.1, 0.2]
    }
    
    best_loss = float('inf')
    best_params = None
    results = []
    
    for lr in param_grid['learning_rate']:
        for hidden in param_grid['hidden_dim']:
            for heads in param_grid['num_heads']:
                for drop in param_grid['dropout']:
                    model = FineTunedModel(hidden_dim=hidden, 
                                         num_heads=heads, 
                                         dropout=drop)
                    model = model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.BCELoss()
                    
                    trained_model = train_with_validation(model, 
                                                        train_loader, 
                                                        val_loader,
                                                        criterion, 
                                                        optimizer, 
                                                        device)
                    
                    val_loss = evaluate_model(trained_model, val_loader, criterion, device)
                    
                    results.append({
                        'params': {
                            'lr': lr,
                            'hidden_dim': hidden,
                            'num_heads': heads,
                            'dropout': drop
                        },
                        'val_loss': val_loss
                    })
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_params = results[-1]['params']
    
    return best_params, results

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for protein_inputs, chemical_inputs, labels in val_loader:
            protein_inputs = {k: v.to(device) for k, v in protein_inputs.items()}
            chemical_inputs = {k: v.to(device) for k, v in chemical_inputs.items()}
            labels = labels.to(device)
            outputs = model(protein_inputs, chemical_inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# %%
df = df.iloc[:20]  # Use only first n rows for demonstration
protein_inputs, chemical_inputs, labels = process_raw_data(df)
dataset = InteractionDataset(protein_inputs, chemical_inputs, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %%
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


# %%
# Train model and grid search
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_params, all_results = grid_search(train_loader, val_loader, device)
print("Best parameters:", best_params)

# %%
# Initialize the optimized cross-attention network with best hyperparameters
optimized_model = FineTunedModel(
    hidden_dim=best_params['hidden_dim'],  # Size of hidden layers
    num_heads=best_params['num_heads'],    # Number of attention heads
    dropout=best_params['dropout']         # Dropout rate for regularization
)

# Create Adam optimizer with optimized learning rate
optimizer = torch.optim.Adam(optimized_model.parameters(), lr=best_params['lr'])

# Move model to GPU/CPU device
optimized_model = optimized_model.to(device)

# %%

# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def calculate_roc_auc(model, test_loader, device):
   #Calculate and plot ROC curve and AUC score for model evaluation
   model.eval()
   all_labels = []
   all_preds = []
   
   # Get predictions
   with torch.no_grad():
       for protein_inputs, chemical_inputs, labels in test_loader:
           # Move inputs to device
           protein_inputs = {k: v.to(device) for k, v in protein_inputs.items()}
           chemical_inputs = {k: v.to(device) for k, v in chemical_inputs.items()}
           
           # Get model predictions
           outputs = model(protein_inputs, chemical_inputs)
           
           # Store predictions and labels
           all_preds.extend(outputs.cpu().numpy())
           all_labels.extend(labels.cpu().numpy())
   
   # Calculate ROC curve points and AUC
   fpr, tpr, _ = roc_curve(all_labels, all_preds)
   auc_score = auc(fpr, tpr)
   
   # Plot ROC curve
   plt.figure()
   plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
   plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('ROC Curve')
   plt.legend()
   plt.show()
   
   return auc_score
# %%
