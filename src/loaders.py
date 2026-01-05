import pandas as pd
import gzip
import os
from config import AUTH_PATH, COLS, CHUNK_SIZE, SEC_IN_DAY, SEC_IN_HOUR, PARQUET_PATH

def auth_stream(start_time=None, stop_time=None):
    """
    Yields chunks of authentication data within a specific time window.
    """
    with gzip.open(AUTH_PATH, "rt") as f:
        for chunk in pd.read_csv(
            f, 
            names=COLS, 
            header=None, 
            chunksize=CHUNK_SIZE
        ):
            # 1. Clean data types
            chunk["Timestamp"] = pd.to_numeric(chunk["Timestamp"], errors='coerce')
            
            # 2. Filter by start_time
            if start_time is not None:
                chunk = chunk[chunk["Timestamp"] >= start_time]
            
            # 3. Filter by stop_time
            if stop_time is not None:
                chunk = chunk[chunk["Timestamp"] < stop_time]
            
            if chunk.empty:
                # If chunk is empty because of filtering, check if we've passed the stop_time entirely
                # Note: This assumes the CSV is temporally sorted (LANL data usually is)
                with gzip.open(AUTH_PATH, "rt") as f_check:
                    # Logic to break if current data is way beyond stop_time
                    pass 
                continue

            # 4. Basic feature labeling
            chunk["IsSuccess"] = (chunk["Success"] == "Success").astype(int)
            
            yield chunk

def get_hour_data_fast(day, hour):
    """
    Retrieves one hour of data in milliseconds using Parquet filters.
    """
    start_ts = ((day - 1) * 86400) + (hour * 3600)
    end_ts = start_ts + 3600
    
    # pyarrow 'filters' do the magic of skipping folders/files
    df = pd.read_parquet(
        PARQUET_PATH,
        filters=[
            ('Day', '==', day),
            ('Timestamp', '>=', start_ts),
            ('Timestamp', '<', end_ts)
        ],
        engine='pyarrow'
    )
    return df