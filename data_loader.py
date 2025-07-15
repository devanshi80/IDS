import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(path, sample_size=10000):
    selected_features = [
        'flow duration',
        'total fwd packets',
        'total backward packets',
        'total length of fwd packets',
        'total length of bwd packets',
        'fwd packet length mean',
        'bwd packet length mean',
        'flow bytes/s',
        'flow packets/s',
        'fwd iat mean',
        'bwd iat mean',
        'packet length mean',
        'packet length std',
        'ack flag count',
        'syn flag count',
        'down/up ratio',
        'average packet size',
        'avg fwd segment size',
        'avg bwd segment size',
        'init_win_bytes_forward',
        'init_win_bytes_backward',
        'act_data_pkt_fwd',
        'idle mean',
        'idle std',
        'label'
    ]

    # Reads full file first
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    print(f" Columns in raw CSV: {list(df.columns)}")

    # Filter only selected ones if they exist
    missing_cols = [col for col in selected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df[selected_features]

    # Clean invalid values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Stratified sampling
    df_sampled, _ = train_test_split(df, train_size=sample_size, stratify=df['label'], random_state=42)

    # Binary encode label
    df_sampled['label'] = df_sampled['label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)

    # Rename for downstream consistency
    df_sampled.rename(columns={'label': 'Label'}, inplace=True)

    return df_sampled
