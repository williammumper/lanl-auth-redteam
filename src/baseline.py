import pickle
import pandas as pd
from config import BASELINE_END_TIME, MIN_LOGINS, MAX_FAIL_RATE, ARTIFACTS_DIR
from loaders import auth_stream

def generate_baseline():
    user_stats = []
    total_rows_processed = 0 # Track progress
    
    print(f"Scanning data for first {BASELINE_END_TIME // 86400} days...")
    
    for chunk in auth_stream(stop_time=BASELINE_END_TIME):
        total_rows_processed += len(chunk)
        
        # We only need the window within the baseline period
        mask = chunk["Timestamp"] <= BASELINE_END_TIME
        batch = chunk[mask]
        
        if not batch.empty:
            # Aggregate stats for this chunk
            agg = batch.groupby("User").agg(
                total_logins=("Timestamp", "count"),
                failed_logins=("IsSuccess", lambda x: (x == 0).sum()),
                unique_comps=("TargetComputer", "nunique")
            )
            user_stats.append(agg)
            
            # Print status update every chunk
            current_time = batch["Timestamp"].max()
            print(f"Processed {total_rows_processed:,} rows... Current Timestamp: {current_time}")
            
    print("\nFinalizing baseline selection...")
    # Combine all chunk-level aggregates
    full_stats = pd.concat(user_stats).groupby("User").sum()
    
    # Calculate global fail rate for these 3 days
    full_stats["fail_rate"] = full_stats["failed_logins"] / full_stats["total_logins"]
    
    # Apply our SOC 'Benign' filters
    baseline_df = full_stats[
        (full_stats["total_logins"] >= MIN_LOGINS) & 
        (full_stats["fail_rate"] <= MAX_FAIL_RATE)
    ]
    
    baseline_users = baseline_df.index.tolist()
    
    # Save the artifact
    output_path = f"{ARTIFACTS_DIR}/baseline_users.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(baseline_users, f)
        
    print(f"Success! {len(baseline_users)} benign users identified and saved to {output_path}")

if __name__ == "__main__":
    generate_baseline()