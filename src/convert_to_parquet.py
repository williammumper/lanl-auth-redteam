import pandas as pd
import os
from config import AUTH_PATH, DATA_DIR, COLS

def convert():
    parquet_dir = os.path.join(DATA_DIR, "auth_parquet")
    os.makedirs(parquet_dir, exist_ok=True)
    
    print("Reading Gzip and converting to Parquet... this will take a moment.")
    
    # Process in large chunks to preserve memory
    reader = pd.read_csv(AUTH_PATH, names=COLS, header=None, chunksize=2_000_000)
    
    for i, chunk in enumerate(reader):
        chunk["Timestamp"] = pd.to_numeric(chunk["Timestamp"], errors='coerce')
        # We create a 'Day' column to use as a partition for even faster lookups
        chunk["Day"] = (chunk["Timestamp"] // 86400) + 1
        
        # Save using 'Day' as a partition folder
        chunk.to_parquet(
            parquet_dir, 
            partition_cols=["Day"], 
            engine="pyarrow", 
            index=False
        )
        print(f"Processed chunk {i+1}")

if __name__ == "__main__":
    convert()