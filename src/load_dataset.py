import gzip
import pandas as pd

AUTH_FILE = "../data/auth.txt.gz"

# Adjust column names to match LANL dataset
COLS = [
    "RecordID", "User", "AccountName", "SourceComputer", 
    "TargetComputer", "AuthPackage", "LogonType", "LogonAction", "Success"
]

def load_auth_sample(nrows=None):
    with gzip.open(AUTH_FILE, "rt") as f:
        df = pd.read_csv(f, sep=",", names=COLS, nrows=nrows, header=None)
    return df

# Load first 10,000 rows
df = load_auth_sample(nrows=10000)

print("Columns:", df.columns.tolist())
print("Head:\n", df.head())

# Basic stats
print("\nBasic stats:")
print("Number of rows:", len(df))
print("Unique users:", df['User'].nunique())
print("Unique computers:", df['TargetComputer'].nunique())
print("Success vs failure counts:\n", df['Success'].value_counts())
