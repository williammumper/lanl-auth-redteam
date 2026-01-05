import pandas as pd
import pickle
import os
from config import ARTIFACTS_DIR, BASELINE_END_TIME
from loaders import auth_stream
from features import extract_features, is_machine

def create_split_training_sets():
    # 1. Load the "Clean User" list from your baseline
    baseline_path = os.path.join(ARTIFACTS_DIR, "baseline_users.pkl")
    with open(baseline_path, "rb") as f:
        benign_users = set(pickle.load(f))
    
    human_chunks = []
    machine_chunks = []
    
    print(f"Splitting data for {len(benign_users)} users into Human and Machine tracks...")
    
    # 2. Re-stream the baseline period (Day 1 & 2)
    for chunk in auth_stream(stop_time=BASELINE_END_TIME):
        # Only keep events for our validated benign users
        filtered_batch = chunk[chunk["User"].isin(benign_users)].copy()
        
        if not filtered_batch.empty:
            # 3. Identify account types using the new regex logic
            mask = is_machine(filtered_batch['User'])
            machines = filtered_batch[mask]
            humans = filtered_batch[~mask]
            
            # 4. Extract features for each group
            if not humans.empty:
                human_chunks.append(extract_features(humans))
            if not machines.empty:
                machine_chunks.append(extract_features(machines))
            
            print(f"Streaming Logs: Timestamp {filtered_batch['Timestamp'].max()}...", end="\r")

    print("\nFinalizing datasets...")

    # 5. Save Human Training Set
    if human_chunks:
        human_df = pd.concat(human_chunks).sort_values(['User', 'Timestamp'])
        human_path = os.path.join(ARTIFACTS_DIR, "training_human.pkl")
        human_df.to_pickle(human_path)
        print(f"Success: Saved {len(human_df):,} human events to {human_path}")

    # 6. Save Machine Training Set
    if machine_chunks:
        machine_df = pd.concat(machine_chunks).sort_values(['User', 'Timestamp'])
        machine_path = os.path.join(ARTIFACTS_DIR, "training_machine.pkl")
        machine_df.to_pickle(machine_path)
        print(f"Success: Saved {len(machine_df):,} machine events to {machine_path}")

if __name__ == "__main__":
    create_split_training_sets()