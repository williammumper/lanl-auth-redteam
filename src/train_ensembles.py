import pandas as pd
import pickle
import os
import gc
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from config import ARTIFACTS_DIR

class AnomalyEnsemble:
    def __init__(self, name):
        self.name = name
        self.scaler = StandardScaler()
        self.models = {
            "iso_forest": IsolationForest(contamination=0.01, random_state=42, n_jobs=-1),
            "oc_svm": OneClassSVM(nu=0.01, kernel="rbf", gamma='auto', verbose=True),
            "lof": LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True, n_jobs=-1)
        }

    def train(self, df_path, rows_per_hour=4000, max_days=7):
        print(f"[{self.name}] Loading data for 7-day stratified sampling...")
        
        # 1. Load data
        df = pd.read_pickle(df_path)
        
        # 2. Filter for the first 7 days (7 * 86,400 seconds)
        day_limit = max_days * 86400
        df = df[df['Timestamp'] <= day_limit].copy()
        
        # 3. Create Hour index for diverse sampling
        # This ensures we see morning, afternoon, and night for every day
        df['HourIdx'] = (df['Timestamp'] // 3600).astype(int)
        
        print(f"[{self.name}] Stratifying: Taking {rows_per_hour} rows/hr...")
        
        # Sample per hour to capture the full weekly rhythm
        df_sampled = df.groupby('HourIdx').apply(
            lambda x: x.sample(n=min(len(x), rows_per_hour), random_state=42)
        ).reset_index(drop=True)
        
        # 4. Feature Selection & Memory Cleanup
        cols_to_use = ['Time_Delta_Log', 'Success_Int', 'LogonType_Int', 'Is_Remote', 'Target_Count']
        X = df_sampled[cols_to_use].astype('float32')
        
        print(f"[{self.name}] Final training set size: {len(X):,}")
        
        del df, df_sampled
        gc.collect()

        # 5. Scaling
        print(f"[{self.name}] Fitting scaler...")
        X_scaled = self.scaler.fit_transform(X)
        
        # 6. Fit Models with 8GB RAM Safety
        for label, model in self.models.items():
            print(f">>> [{self.name}] Fitting {label}...")
            
            if label == "oc_svm":
                # One-Class SVM is O(N^2) memory. Cap it at 150k rows.
                # This still gives it a great representation of the 7-day window.
                model.fit(X_scaled[:150000])
            else:
                # IsoForest and LOF can handle the full 672k rows easily.
                model.fit(X_scaled)
                
            gc.collect() 
        
        print(f"[{self.name}] Training Complete.")

    def save(self):
        path = os.path.join(ARTIFACTS_DIR, f"ensemble_{self.name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[{self.name}] Ensemble saved to {path}")

def main():
    paths = {
        "human": os.path.join(ARTIFACTS_DIR, "training_human.pkl"),
        "machine": os.path.join(ARTIFACTS_DIR, "training_machine.pkl")
    }

    for track, path in paths.items():
        if os.path.exists(path):
            print(f"\n{'='*20} TRACK: {track.upper()} {'='*20}")
            ens = AnomalyEnsemble(name=track)
            
            # 7 Days at 4k rows per hour = ~672,000 rows total
            ens.train(path, rows_per_hour=4000, max_days=7)
            ens.save()
            
            del ens
            gc.collect()
        else:
            print(f"Skipping {track}, file not found.")

if __name__ == "__main__":
    main()