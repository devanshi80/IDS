# src/model_trainer.py
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

def train_and_save_model(df, model_path):
    X = df.drop("Label", axis=1)
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define individual classifiers
    clf1 = LogisticRegression(max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf3 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    clf4 = MLPClassifier(max_iter=300)
    clf5 = SVC(probability=True)

    # Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('lr', clf1),
            ('rf', clf2),
            ('xgb', clf3),
            ('mlp', clf4),
            ('svc', clf5)
        ],
        voting='soft'  # Use soft voting for probabilities
    )

    voting_clf.fit(X_train, y_train)

    # Evaluate
    y_pred = voting_clf.predict(X_test)
    print("Voting Classifier Report:\n", classification_report(y_test, y_pred))

    # Save ensemble model
    with open(model_path, 'wb') as f:
        pickle.dump(voting_clf, f)

    return voting_clf
