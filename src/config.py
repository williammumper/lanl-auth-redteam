import os

# --- PATHS ---
DATA_DIR = "./data"
ARTIFACTS_DIR = "./artifacts"
AUTH_PATH = os.path.join(DATA_DIR, "auth.txt.gz")
REDTEAM_PATH = os.path.join(DATA_DIR, "redteam.txt.gz")
PARQUET_PATH = os.path.join(DATA_DIR, "auth_parquet")

# Create artifacts dir if it doesn't exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- CONSTANTS ---
SEC_IN_DAY = 86400
SEC_IN_HOUR = 3600
BASELINE_DAYS = 2
BASELINE_END_TIME = SEC_IN_DAY * BASELINE_DAYS

# --- DATAFRAME CONFIG ---
CHUNK_SIZE = 500_000
COLS = [
    "Timestamp", "User", "AccountName", "SourceComputer", 
    "TargetComputer", "AuthPackage", "LogonType", 
    "LogonAction", "Success"
]

# --- BASELINE CRITERIA ---
MIN_LOGINS = 20
MAX_FAIL_RATE = 0.01