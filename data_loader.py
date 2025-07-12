import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # clean column names
    return df

def preprocess_dataset(df):
    df = df.dropna()
    df = df.drop(columns=["Timestamp"], errors='ignore')
    
    # Convert labels to binary if multiclass
    if 'Label' in df.columns:
        df['Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
    
    # Encode categorical columns
    for col in df.select_dtypes(include='object'):
        df[col] = pd.factorize(df[col])[0]
    
    return df
