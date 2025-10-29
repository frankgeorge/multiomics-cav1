# Multi-omics model training script (example)
# This is an example training script. For heavy training, use Colab or a GPU machine.
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('int64')
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

def main():
    # expects normalized_counts.csv and merged_tcga_data.csv in working dir
    if not os.path.exists('normalized_counts.csv') or not os.path.exists('merged_tcga_data.csv'):
        print('Please run the R preprocessing script first (R/tcga_preprocessing.R) and place normalized_counts.csv and merged_tcga_data.csv here.')
        return
    expr = pd.read_csv('normalized_counts.csv', index_col=0)
    merged = pd.read_csv('merged_tcga_data.csv')
    # transpose expr -> samples x genes
    expr_t = expr.T
    df = expr_t.merge(merged, left_index=True, right_on='sample_id', how='left')
    # create a demo proxy label from CAV1 expression
    if 'Cav1_expression' not in df.columns:
        print('CAV1 expression not found in merged file; labels will be random (demo).')
        df['Cav1_expression'] = np.random.randn(len(df))
    y = (df['Cav1_expression'] > np.nanmedian(df['Cav1_expression'])).astype(int).values
    X = df.drop(columns=['sample_id','Cav1_expression'])
    X = X.fillna(0).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_ds = SimpleDataset(X_train, y_train)
    test_ds = SimpleDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    for epoch in range(1,6):
        model.train()
        total=0
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f'Epoch {epoch} loss {total/len(train_loader):.4f}')
    # save model state
    os.makedirs('models', exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'input_dim': X.shape[1]}, 'models/cav1_model.pth')
    print('Model saved to models/cav1_model.pth')

if __name__ == '__main__':
    main()
