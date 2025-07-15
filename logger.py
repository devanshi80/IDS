import csv
import os

def log_risk_score(input_dict, prediction, probability, log_path):
    """
    Logs input values, prediction, and probability to CSV.
    Creates the file with headers if it doesn’t exist.
    """
    fieldnames = list(input_dict.keys()) + ['Prediction', 'Probability']

    log_exists = os.path.isfile(log_path)

    with open(log_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not log_exists:
            writer.writeheader()

        row = input_dict.copy()
        row['Prediction'] = prediction
        row['Probability'] = round(probability, 4)

        writer.writerow(row)

