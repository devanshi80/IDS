import pandas as pd
import numpy as np
import joblib

def predict_single_input(input_dict, model_path):
    """
    Predict on a single input dictionary.
    """
    model = joblib.load(model_path)

    # Ensure consistent column order
    feature_order = model.feature_names_in_

    # Convert dict to DataFrame
    df = pd.DataFrame([input_dict])

    # Reorder + fill any missing with 0
    df = df.reindex(columns=feature_order, fill_value=0)

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return prediction, probability

def predict_batch_inputs(df, model_path):
    """
    Predict on a batch dataframe.
    """
    model = joblib.load(model_path)
    
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])

    # Ensure correct column order
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    return pd.DataFrame({
        'Prediction': preds,
        'Probability': probs
    })
