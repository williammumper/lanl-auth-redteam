import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm # You may need to: pip install tqdm
from torch.utils.data import DataLoader, TensorDataset
from models import LSTMAutoencoder
from config import ARTIFACTS_DIR

def train_track_safe(name, file_path, epochs=3, chunk_size=500000, window_size=20):
    print(f"\n{'='*20} TRAINING: {name.upper()} {'='*20}")
    device = torch.device("cpu")
    
    model = LSTMAutoencoder(input_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 1. Load and immediately downcast
    print(f"[{name}] Loading data into RAM...")
    full_df = pd.read_pickle(file_path)
    data_np = full_df.iloc[:, 2:].values.astype('float32')
    
    # Total sequences to process
    total_samples = len(data_np)
    
    del full_df 
    gc.collect()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_chunks = 0
        
        # tqdm progress bar wrapper
        pbar = tqdm(total=total_samples, desc=f"Epoch {epoch+1}/{epochs}", unit="rows")
        
        for i in range(0, total_samples, chunk_size):
            chunk = data_np[i : i + chunk_size]
            num_seq = len(chunk) // window_size
            if num_seq == 0: continue
                
            X = chunk[:num_seq * window_size].reshape(num_seq, window_size, -1)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            
            loader = DataLoader(TensorDataset(X_tensor, X_tensor), batch_size=512, shuffle=True)
            
            for batch_x, _ in loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_chunks += 1
            
            # Update progress bar by the chunk size
            pbar.update(len(chunk))
            
        pbar.close()
        print(f">>> {name.upper()} Epoch {epoch+1} Avg Loss: {epoch_loss/num_chunks:.6f}")

    # Save
    save_path = os.path.join(ARTIFACTS_DIR, f"lstm_{name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[{name}] Model saved.")
    
    del data_np
    gc.collect()

def main():
    # Only proceed if the artifacts exist
    tracks = [
        ("human", os.path.join(ARTIFACTS_DIR, "training_human.pkl")),
        ("machine", os.path.join(ARTIFACTS_DIR, "training_machine.pkl"))
    ]

    for name, path in tracks:
        if os.path.exists(path):
            train_track_safe(name, path)
        else:
            print(f"Skipping {name}: file not found.")

if __name__ == "__main__":
    main()