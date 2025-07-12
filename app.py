# app.py
from src.data_loader import load_dataset, preprocess_dataset
from src.model_trainer import train_and_save_model
from src.predictor import load_model, predict
from src.port_scanner import scan_ports, build_feature_vector
from src.utils import map_port_to_feature

import os

DATA_PATH = "data/Friday-WorkingHours-Morning.pcap_ISCX.csv"
MODEL_PATH = "models/best_model.pkl"

def run_training():
    df = load_dataset(DATA_PATH)
    df = preprocess_dataset(df)
    train_and_save_model(df, MODEL_PATH)

def run_detection():
    model = load_model(MODEL_PATH)
    ports = scan_ports()
    print(f"Open ports: {ports}")
    for port in ports:
        features = {"port": port}  # Simplified
        label = predict(model, features)
        print(f"Port {port} → {'BENIGN' if label == 0 else 'BOT-LIKE'}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Threat Detection System")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--scan", action="store_true", help="Run port scan + threat detection")
    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.scan:
        run_detection()
    else:
        print("Use --train or --scan")
