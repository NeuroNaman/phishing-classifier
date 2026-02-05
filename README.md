# 🛡️ End-to-End Phishing URL Detection System  
### Production-Grade Machine Learning System with Flask, MongoDB, MLOps Pipelines & Docker

---

## 📌 Executive Summary

Phishing attacks remain one of the most common and damaging cybersecurity threats, exploiting human trust through malicious URLs embedded in emails, messages, and fake websites.

This project is a **full-stack, production-oriented Phishing URL Detection System** that:

- Detects whether a URL is **phishing or legitimate**
- Implements a **complete ML lifecycle**
- Uses **clean architecture and modular pipelines**
- Integrates **Flask for inference delivery**
- Stores and ingests data from **MongoDB**
- Is **Dockerized** and **CI/CD-ready**
- Follows **industry MLOps and software engineering practices**

This is **not a notebook-only ML project**.  
It is designed as a **deployable, scalable system**.

---

## 🎯 Problem Statement

Phishing websites mimic trusted domains to steal credentials, financial data, and personal information.

Traditional rule-based systems fail because:
- Attack patterns evolve rapidly
- Manual rules do not scale
- Static blacklists become obsolete

### Solution
Use **machine learning** to classify URLs using:
- URL structure features
- Domain-based signals
- SSL and security indicators
- Redirection and behavioral patterns
- Statistical and reputation-based features

---

## 🧠 Key Design Goals

- End-to-end automation
- Separation of concerns
- Reproducibility
- Config-driven ML pipelines
- Production-grade inference
- Interview-ready architecture

## 📊 Exploratory Data Analysis (EDA)

![alt text](image.png)

EDA Insights:

URL-based indicators are dominant

Binary features strongly influence predictions

Behavioral and domain signals improve accuracy

📈 Feature Categories
Category	Examples
URL Structure	Length, IP usage, symbols
Domain Info	Age, DNS record
Security	SSL state, HTTPS token
Behavior	Redirects, popups
Reputation	Traffic, page rank

## 🌐 Flask Application Layer

Flask acts as a **thin orchestration layer**.

### Routes

| Route | Method | Description |
|-----|------|------------|
| `/` | GET | Load prediction UI |
| `/train` | GET | Trigger full training pipeline |
| `/predict` | POST | Upload CSV and download predictions |

No ML logic is written inside Flask routes.

---

## 🔁 Training Pipeline

**File:** `src/pipeline/train_pipeline.py`

Execution order:

run_pipeline()
├── Data Ingestion
├── Data Validation
├── Data Transformation
└── Model Training


Each stage is:
- Independent
- Logged
- Exception-safe
- Artifact-producing

---

## 📥 Data Ingestion

**Source:** MongoDB  
**File:** `data_ingestion.py`

Responsibilities:
- Connect to MongoDB
- Export collections
- Save raw CSV files to artifacts directory

---

## ✅ Data Validation

**File:** `data_validation.py`  
**Config:** `training_schema.json`

Validations:
- File name pattern
- Timestamp format
- Column count
- Missing value checks

Validated and invalid data are separated physically.

---

## 🔄 Data Transformation

**File:** `data_transformation.py`

Steps:
- Merge validated batches
- Remove unwanted spaces
- Handle missing values
- Encode target labels
- Handle class imbalance (RandomOverSampler)
- Train-test split
- Save preprocessing object

---

## 🧠 Model Training & Selection

**File:** `model_trainer.py`

Models evaluated:
- Logistic Regression
- Gaussian Naive Bayes
- XGBoost Classifier

Process:
1. Train all models
2. Compare accuracy
3. Select best model
4. Hyperparameter tuning using GridSearchCV
5. Final training
6. Save model artifact

All hyperparameters are defined in `config/model.yaml`.

---

## 📦 Model Artifact Design

The saved model includes:
- Preprocessing object
- Trained ML model

This ensures **training–inference consistency**.

---

## 🔮 Prediction Pipeline

**File:** `predict_pipeline.py`

Flow:
1. CSV upload
2. Save input file
3. Load model + preprocessor
4. Transform features
5. Generate predictions
6. Save output CSV
7. Download predictions

🪵 Logging & Observability

Centralized logging with:

Timestamped logs

File-based storage

Structured messages

Enables debugging and production monitoring.

❌ Exception Handling

Custom exception system captures:

File name

Line number

Error message

Full traceback

🐳 Dockerization

Dockerfile:

FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python3","app.py"]


Benefits:

Reproducible environment

Easy deployment

Cloud ready

🚀 Run Locally
git clone https://github.com/your-username/phishing-classifier.git
cd phishing-classifier
pip install -r requirements.txt
python app.py


Open:
http://127.0.0.1:5050

🧪 Sample Input
having_IP_Address,URL_Length,...
1,54,...

📤 Output
...,Result
...,phishing
...,safe

🧠 Interview Talking Points

Modular ML pipelines

Artifact-driven workflows

Flask orchestration

Config-driven modeling

Data validation strategies

Dockerized ML systems

🔮 Future Enhancements

MLflow integration

Async training

REST APIs

Cloud deployment

Monitoring dashboards

Drift detection

⭐ Why This Project Matters

This repository demonstrates:

Real-world ML engineering

Production architecture

MLOps best practices

Interview-ready system design

🤝 Contributing

Fork → Create branch → Submit PR 🚀

⭐ Star This Repo

If you found this useful, give it a ⭐
It helps others discover quality ML projects.



'''pip install -e .  ''' to run setup.py
