import pandas as pd
import os
from datetime import datetime

LOG_PATH = "logs/risk_scores.csv"

def log_prediction(port, prob, label):
    risk = (
        "Critical" if prob >= 0.9 else
        "High" if prob >= 0.7 else
        "Medium" if prob >= 0.4 else
        "Low"
    )
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "port": port,
        "probability": round(prob * 100, 2),
        "label": "BOT" if label == 1 else "BENIGN",
        "risk": risk
    }

    if not os.path.exists(LOG_PATH):
        df = pd.DataFrame([log_data])
    else:
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame([log_data])], ignore_index=True)

    df.to_csv(LOG_PATH, index=False)
