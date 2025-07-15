import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def train_and_save_model(df, model_path):
    # Separate features and label
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Individual models
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Train individual models
    logreg.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)             # Tree-based: unscaled
    mlp.fit(X_train_scaled, y_train)
    xgb.fit(X_train, y_train)

    # Voting Classifier: mix scaled and unscaled where appropriate
    voting_clf = VotingClassifier(estimators=[
        ('logreg', logreg),
        ('mlp', mlp),
        ('xgb', xgb)
    ], voting='soft')

    voting_clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = voting_clf.predict(X_test_scaled)
    print("\n Classification Report:\n", classification_report(y_test, y_pred))
    print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model and scaler
    joblib.dump((voting_clf, scaler), model_path)
    print(f" VotingClassifier saved to {model_path}")

