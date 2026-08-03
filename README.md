# 📦 Vendor Invoice Intelligence Portal

An end-to-end **Machine Learning and Analytics** system designed to streamline financial operations by predicting freight shipping costs and automatically identifying high-risk vendor invoices that require manual review.

---

## 📌 Project Overview

Large retail organizations process thousands of vendor invoices every day. Manual verification is slow, expensive, and susceptible to human error.

The **Vendor Invoice Intelligence Portal** automates invoice auditing using Machine Learning through two core modules:

### 🚚 Freight Cost Prediction
A regression model that forecasts expected freight shipping costs based on invoice information, helping organizations:

- Improve budgeting
- Detect unusual freight charges
- Support vendor negotiations
- Optimize logistics planning

### 🚨 Invoice Risk Flagging
A classification model that analyzes invoice discrepancies and delivery patterns to determine whether an invoice should:

- ✅ Be Auto Approved
- ⚠️ Require Manual Review

The model identifies risky invoices using historical procurement and receiving data.

---

# ✨ Features

- 📈 Freight Cost Prediction using Regression
- 🚨 Intelligent Invoice Risk Detection
- 📊 Interactive Streamlit Dashboard
- 🗄️ SQLite Database Integration
- 📉 Data Visualization & Analytics
- 🤖 Automated Machine Learning Pipelines
- 💾 Serialized Models using Joblib

---

# 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Dashboard | Streamlit |
| Database | SQLite |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Models | Random Forest, Linear Regression, Decision Tree |
| Model Serialization | Joblib |

---

# 📂 Project Structure

```text
Vendor Invoice Intelligence Portal/
│
├── data/
│   └── inventory.db
│
├── models/
│   ├── predict_freight_model.pkl
│   ├── predict_flag_invoice.pkl
│   └── scaler.pkl
│
├── Freight Cost Prediction/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   └── train.py
│
├── Invoice Flagging/
│   ├── data_preprocessing.py
│   ├── modeling_evaluation.py
│   └── train.py
│
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── notebook/
│   ├── predicting_freight_cost.ipynb
│   └── invoice_flagging.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🚚 Module 1: Freight Cost Prediction

## 🎯 Objective

Predict the expected freight cost associated with vendor invoices to assist procurement and logistics teams.

### Input Features

- Invoice Quantity
- Invoice Dollar Amount
- Total Item Quantity
- Total Item Dollar Amount

### Target

- Freight Cost

### Machine Learning Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Best Performing Model

**Random Forest Regressor**

---

# 🚨 Module 2: Invoice Risk Flagging

## 🎯 Objective

Automatically detect invoices that are potentially fraudulent, erroneous, or delayed.

### Input Features

- Invoice Quantity
- Invoice Dollars
- Freight Cost
- Total Item Quantity
- Total Item Dollars

### Target Variable

```text
flag_invoice

1 → Manual Approval Required

0 → Auto Approved
```

---

## 📋 Invoice Flagging Rules

An invoice is flagged if **any** of the following conditions are met:

- Invoice amount differs from total item amount by more than **$5**
- Average receiving delay exceeds **10 days**

These business rules were used to generate labels for supervised learning.

---

## Machine Learning Pipeline

1. Data Cleaning
2. Feature Engineering
3. Feature Scaling using StandardScaler
4. Model Training
5. Hyperparameter Tuning
6. Prediction
7. Model Serialization

---

## Best Model

**Random Forest Classifier**

Feature Scaling:

- StandardScaler

---

# 📊 Dashboard

The project includes an interactive **Streamlit Dashboard** that allows users to:

- Predict Freight Costs
- Detect High-Risk Invoices
- View Prediction Results
- Perform Real-Time Inference
- Analyze Invoice Data

---

# 🚀 Getting Started

## Prerequisites

- Python 3.10+
- pip

---

## Clone Repository

```bash
git clone https://github.com/your-username/vendor-invoice-intelligence-portal.git

cd vendor-invoice-intelligence-portal
```

---

## Install Dependencies

```bash
pip install pandas numpy scikit-learn streamlit joblib
```

or

```bash
pip install -r requirements.txt
```

---

# 🧠 Train the Models

## Freight Cost Prediction

```bash
cd "Freight Cost Prediction"

python train.py

cd ..
```

---

## Invoice Risk Flagging

```bash
cd "Invoice Flagging"

python train.py

cd ..
```

---

# ▶️ Run the Dashboard

```bash
streamlit run app.py
```

Once the application starts, open your browser and navigate to:

```
http://localhost:8501
```

---

# 📈 Machine Learning Workflow

```text
SQLite Database
        │
        ▼
Data Extraction
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Feature Scaling
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Save Model (.pkl)
        │
        ▼
Streamlit Dashboard
        │
        ▼
Real-Time Predictions
```

---

# 📦 Saved Models

| Model | Purpose |
|--------|---------|
| predict_freight_model.pkl | Freight Cost Prediction |
| predict_flag_invoice.pkl | Invoice Risk Classification |
| scaler.pkl | Feature Normalization |

---

# 📚 Libraries Used

- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- SQLite3
- Joblib

---

# 📌 Future Improvements

- Deep Learning-based anomaly detection
- XGBoost and LightGBM integration
- Explainable AI using SHAP
- Real-time database connectivity
- Role-based authentication
- Cloud deployment (AWS/Azure)
- REST API integration with FastAPI
- Interactive business dashboards with Power BI

---

# 👨‍💻 Author

**Shivendu Kumar**

- B.Tech CSE (AI & ML)
- Full Stack & Machine Learning Developer

---

# ⭐ If you found this project useful, consider giving it a star on GitHub!