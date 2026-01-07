import torch
import pandas as pd
import pickle
import os
import numpy as np
from config import ARTIFACTS_DIR, REDTEAM_PATH
from features import extract_features, is_machine
from models import LSTMAutoencoder


def gather_windows(X, indices, window):
    valid = indices[indices >= window - 1]
    if len(valid) == 0:
        return None, None
    windows = np.stack([X[i - window + 1:i + 1] for i in valid])
    return windows, valid


class SOCSimulator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.tracks = ["human", "machine"]
        self.ensembles = {}
        self.lstms = {}
        self.optimizers = {}
        self.criterion = torch.nn.MSELoss()
        self.window_size = 20

        # Stateful buffers
        self.buffers = {t: pd.DataFrame() for t in self.tracks}

        for t in self.tracks:
            ensemble_path = os.path.join(ARTIFACTS_DIR, f"ensemble_{t}.pkl")
            with open(ensemble_path, "rb") as f:
                self.ensembles[t] = pickle.load(f)

            model = LSTMAutoencoder(input_dim=5).to(self.device)
            lstm_path = os.path.join(ARTIFACTS_DIR, f"lstm_{t}.pth")
            if os.path.exists(lstm_path):
                model.load_state_dict(torch.load(lstm_path, map_location=self.device))

            model.eval()
            self.lstms[t] = model
            self.optimizers[t] = torch.optim.Adam(model.parameters(), lr=1e-5)

    def verify_red_team(self, hour_alerts):
        if hour_alerts.empty or not os.path.exists(REDTEAM_PATH):
            return pd.DataFrame()

        red_df = pd.read_csv(
            REDTEAM_PATH,
            names=["Timestamp", "User", "Src", "Dst"],
            header=None
        )
        red_df["Timestamp"] = red_df["Timestamp"].astype(int)

        return pd.merge(hour_alerts, red_df, on=["Timestamp", "User"], how="inner")

    def process_hour_stream(self, df, progress_bar=None):
        if df.empty:
            return pd.DataFrame()

        is_mach_mask = is_machine(df["User"])
        splits = {
            "human": df[~is_mach_mask].copy(),
            "machine": df[is_mach_mask].copy()
        }

        all_track_alerts = []

        for t in self.tracks:
            track_df = splits[t]
            if track_df.empty:
                continue

            combined_df = pd.concat(
                [self.buffers[t], track_df],
                axis=0
            ).sort_values("Timestamp")

            X_raw = extract_features(combined_df)
            feature_cols = [
                "Time_Delta_Log",
                "Success_Int",
                "LogonType_Int",
                "Is_Remote",
                "Target_Count"
            ]

            X_scaled = self.ensembles[t].scaler.transform(
                X_raw[feature_cols]
            ).astype("float32")

            if progress_bar:
                progress_bar.progress(0.2, f"{t}: features extracted")

            iso = self.ensembles[t].models["iso_forest"]
            iso_scores = iso.score_samples(X_scaled)
            iso_flag = iso.predict(X_scaled) == -1

            TOP_K = 10000
            candidate_idx = np.argsort(iso_scores)[:TOP_K]

            windows, valid_idx = gather_windows(
                X_scaled, candidate_idx, self.window_size
            )

            lstm_errors = np.zeros(len(X_scaled))

            if windows is not None:
                with torch.no_grad():
                    X_tensor = torch.from_numpy(windows).to(self.device)
                    output = self.lstms[t](X_tensor)
                    errs = ((output[:, -1, :] - X_tensor[:, -1, :]) ** 2).mean(dim=1)
                    lstm_errors[valid_idx] = errs.cpu().numpy()

            if progress_bar:
                progress_bar.progress(0.7, f"{t}: LSTM complete")

            evaluated = lstm_errors[lstm_errors > 0]
            thresh = np.percentile(evaluated, 99) if len(evaluated) else 1.0
            lstm_flag = lstm_errors > thresh

            alert_mask = iso_flag | lstm_flag

            combined_df["priority_score"] = (
                iso_flag.astype(int) + lstm_flag.astype(int)
            ) / 2.0
            combined_df["lstm_error"] = lstm_errors
            combined_df["track"] = t

            new_data_start = len(self.buffers[t])  # buffer rows    
            current_alerts = combined_df.iloc[new_data_start:][alert_mask[new_data_start:]]
            if not current_alerts.empty:
                all_track_alerts.append(current_alerts)

            self.buffers[t] = combined_df.tail(self.window_size - 1).copy()

            if progress_bar:
                progress_bar.progress(1.0, f"{t}: alerts scored")

        return pd.concat(all_track_alerts) if all_track_alerts else pd.DataFrame()

    def _evolve(self, track, clean_data):
        model = self.lstms[track]
        opt = self.optimizers[track]
        model.train()

        sample_size = min(len(clean_data) // self.window_size, 32)
        if sample_size <= 0:
            return

        indices = np.random.choice(
            len(clean_data) - self.window_size,
            sample_size,
            replace=False
        )

        X_batch = np.array([
            clean_data[i:i + self.window_size] for i in indices
        ])

        X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(self.device)

        opt.zero_grad()
        output = model(X_tensor)
        loss = self.criterion(output, X_tensor)
        loss.backward()
        opt.step()

        model.eval()
        del X_tensor, output
