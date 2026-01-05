import pandas as pd
import numpy as np
import os

# Static mappings for consistency across all models
LOGON_TYPES = {
    'Network': 1, 'Interactive': 2, 'Batch': 3, 
    'Service': 4, 'Unlock': 5, 'NetworkCleartext': 6, 
    'NewCredentials': 7, 'RemoteInteractive': 8, 'CachedInteractive': 9
}

def is_machine(user):
    """
    Returns True if the user is a machine account (ends in $).
    Works for both single strings and pandas Series.
    """
    if isinstance(user, str):
        # Handle single string (for .apply)
        return '$' in user
    # Handle pandas Series (for vector operations)
    return user.str.contains(r'\$', na=False)

def extract_features(df):
    """
    Transforms raw logs into numerical features. 
    Note: We do NOT drop 'User' or 'Timestamp' here so the generator can sort them.
    """
    # Ensure chronological order per user for valid time deltas
    df = df.sort_values(['User', 'Timestamp'])
    
    # Binary/Categorical Features
    df['Success_Int'] = (df['Success'] == 'Success').astype(int)
    df['LogonType_Int'] = df['LogonType'].map(lambda x: LOGON_TYPES.get(x, 0))
    df['Is_Remote'] = (df['SourceComputer'] != df['TargetComputer']).astype(int)
    
    # Temporal Features
    df['Time_Delta'] = df.groupby('User')['Timestamp'].diff().fillna(0)
    df['Time_Delta_Log'] = np.log1p(df['Time_Delta'])
    
    # Volumetric/Behavioral Features
    df['Target_Count'] = df.groupby('User')['TargetComputer'].transform('nunique')
    
    feature_cols = [
        'User', 'Timestamp', 'Time_Delta_Log', 'Success_Int', 
        'LogonType_Int', 'Is_Remote', 'Target_Count'
    ]
    return df[feature_cols]