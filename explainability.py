import shap
import pandas as pd
import matplotlib.pyplot as plt

def explain_prediction(model, input_features, feature_names=None):
    shap.initjs()

    X = pd.DataFrame([input_features])
    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X)

    fig = shap.plots.waterfall(shap_values[0], show=False)
    return fig
