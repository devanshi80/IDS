# 🛡️ Smart Threat Detection

A real-time, ML-powered system to detect potentially malicious bot-related activity by scanning open network ports on your machine. Built with ensemble learning, SHAP explainability, and an interactive Streamlit UI.

---

## 🚀 Project Overview

Smart Threat Detection is an intelligent threat monitoring tool that scans your local machine for open ports and classifies each port as **benign** or **bot-related** using a trained ensemble ML model. It combines insights from five popular classifiers using a soft-voting strategy and provides **transparent, explainable results** using SHAP (SHapley Additive exPlanations).

Built to demonstrate full-stack machine learning + system-level monitoring for Major League Hacking (MLH) Fellowship application.

---

##Key Features

-  **VotingClassifier** combining:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - MLP Classifier
  - Support Vector Machine (SVM)

- 🔍 **Real-Time Port Scanning** using `psutil`
- 🧠 **Model Explainability** via SHAP waterfall plots
- 🌐 **Interactive Streamlit UI** for port scanning, CSV uploads, and visual feedback
- 📈 **Risk Scoring System** with automatic logging to CSV
- 🧪 **CLI Interface** for training and scanning
- 🔧 **Modular Codebase** for easy extension and maintenance

---

## 📁 Project Structure

```

smart\_threat\_detection/
│
├── data/                        # Dataset (CICIDS2017)
├── models/                      # Trained VotingClassifier (best\_model.pkl)
├── logs/                        # Logs of risk scores per scan
├── src/                         # Modular Python source code
│   ├── data\_loader.py           # Load and preprocess dataset
│   ├── model\_trainer.py         # Train VotingClassifier
│   ├── predictor.py             # Inference utilities
│   ├── port\_scanner.py          # Real-time open port detection
│   ├── explainability.py        # SHAP explainability
│   ├── logger.py                # Log risk scores
│   └── utils.py                 # Helper functions
│
├── app.py                       # CLI entry point
├── streamlit\_app.py             # Streamlit UI
├── requirements.txt             # Python dependencies
└── README.md                    # This file

````

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/SmartThreatDetection.git
cd SmartThreatDetection
pip install -r requirements.txt
````

---

## 🖥️ Run the Streamlit App (UI)

```bash
streamlit run streamlit_app.py
```

**Features:**

* Scan your system for open ports.
* View classification results and confidence scores.
* Understand predictions using SHAP waterfall plots.
* Upload a `.csv` file with ports for batch predictions.
* View and filter historical predictions from log file.

---

## 🧪 Run via CLI

### 🔹 Train the Model

```bash
python app.py --train
```

### 🔹 Scan & Classify Ports

```bash
python app.py --scan
```

---

## 📊 Risk Score Logging

Every prediction is automatically logged to `logs/risk_scores.csv` with:

* Timestamp
* Port number
* Prediction label (BENIGN / BOT)
* Probability score
* Risk level (Low, Medium, High, Critical)

---

## 🔍 SHAP Explainability

Uses SHAP to visualize feature contributions for each prediction, increasing trust and transparency in the ensemble model.

* SHAP waterfall plots are displayed directly in the Streamlit app
* Great for interviews and interpretability

---

## 📈 Example Screenshot

> *(Optional: Insert screenshot or gif of Streamlit UI or SHAP output here)*

---

## 🛡️ Dataset Used

* **CICIDS2017** – Friday-WorkingHours Morning subset
* Preprocessed and mapped: `BENIGN` = 0, others = 1 (bot-like)

---

## 📌 Dependencies

* `scikit-learn`
* `xgboost`
* `shap`
* `psutil`
* `pandas`, `numpy`
* `matplotlib`, `seaborn`
* `streamlit`

Install all with:

```bash
pip install -r requirements.txt
```

---

## 👩‍💻 Author

**Devanshi Jain**

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙌 Acknowledgments

* CICIDS2017 Dataset - Canadian Institute for Cybersecurity
* SHAP - Explainable AI Toolkit
* Streamlit - ML dashboard framework
* MLH Fellowship team for the opportunity to showcase this work

---
