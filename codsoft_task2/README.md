# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using Logistic Regression, Decision Trees, and Random Forests.
This project includes data preprocessing, feature engineering, model training, threshold tuning, and a Streamlit web app for real-time fraud prediction.

---

## 📌 Project Overview

Credit card fraud is a major challenge impacting customers and businesses worldwide.
This project builds a machine learning model capable of classifying credit card transactions as **fraudulent** or **legitimate**.

✔ Multiple ML models built and compared

✔ Full data preprocessing pipeline

✔ Advanced feature engineering

✔ Streamlit web app for live prediction

✔ Best model saved for deployment


---

## 📂 Dataset

Uses two files provided in the internship task:

* `fraudTrain.csv` (training data)
* `fraudTest.csv` (test data)

Each transaction includes:

* Timestamp
* Merchant details
* Category
* Amount
* Customer demographics
* GPS coordinates
* Target label: `is_fraud`

---

## 🛠️ Tech Stack

* Python 3
* NumPy, Pandas
* Scikit-Learn
* Imbalanced-Learn
* Joblib
* Streamlit

---

## ⚙️ Features & Pipeline

### ✔ Data Preprocessing

* Removed unnecessary identifiers
* Parsed timestamps
* Handled missing values
* One-hot encoded categorical features

### ✔ Feature Engineering

* Customer age
* Transaction hour, day, month, weekday
* Haversine distance (customer ↔ merchant)
* Unix timestamp

### ✔ Handling Imbalance

* Random undersampling
* Class weights
* Used PR-AUC and Recall as key metrics

---

## 🤖 Models Trained

| Model                    | ROC-AUC    | PR-AUC     | Fraud Recall | Notes                      |
| ------------------------ | ---------- | ---------- | ------------ | -------------------------- |
| Logistic Regression      | 0.8709     | 0.0977     | 0.3436       | Weak baseline              |
| Decision Tree            | 0.9590     | 0.5279     | 0.9487       | High recall, low precision |
| **Random Forest (BEST)** | **0.9851** | **0.6756** | **0.9016**   | Best overall               |

🏆 **Random Forest selected as final model.**

---

## 📊 Evaluation Metrics (Random Forest)

* ROC-AUC: **0.9851**
* PR-AUC: **0.6756**
* Recall (Fraud): **0.9016**
* Precision (Fraud): **0.1432**
* Accuracy: **0.9788**

Focus is on **recall**, because missing fraud is more costly than false alarms.

---

## 🚦 Threshold Tuning

Fraud detection uses probability → threshold controls sensitivity.

| Threshold      | Precision | Recall    | Use-Case            |
| -------------- | --------- | --------- | ------------------- |
| 0.30           | Low       | Very High | Max fraud catch     |
| 0.50 (default) | Balanced  | High      | General             |
| 0.70           | High      | Moderate  | Reduce false alerts |

---

## 🏗️ Project Structure

```
credit-card-fraud/
│── data/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
│
│── models/
│   └── best_model_rf.joblib
│
│── app.py                # Streamlit web app
│── fraud_detection.py    # All-in-one ML pipeline
│── requirements.txt
│── README.md
```

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train the model

```
python fraud_detection.py train
```

### 3. Predict using CLI

```
python fraud_detection.py predict --rows sample.json --threshold 0.5
```

---

## 🌐 Running the Streamlit App

```
streamlit run app.py
```

Features:

* Interactive fraud prediction
* Probability output
* Threshold slider
* Feature importance sidebar

---

## 📝 Sample Input (Use in Streamlit Form)

**Legit Transaction Example:**

```
Datetime: 2019-12-08 10:05:25
Merchant: coffee_shop
Category: food_dining
Amount: 8.75
Gender: M
DOB: 1996-04-10
Job: Software Engineer
Lat: 40.7128
Lon: -74.0060
City Pop: 8419000
Merchant Lat: 40.7142
Merchant Lon: -74.0059
```

---

## 📌 Results

* High recall fraud model
* Deployable via Streamlit
* Robust engineered features
* Tunable for different risk tolerances

---

## 🚀 Conclusion

This project fully meets the requirements of the internship task:

✔ Built multiple models

✔ Compared using proper metrics

✔ Engineered strong features

✔ Handled imbalance

✔ Delivered a real-time fraud detector app


---

## Important Notice: Model File

The trained model file (`best_model_rf.joblib`) is 113MB and too large to be uploaded to GitHub.

**You can download the required `best_model_rf.joblib` file from this link:**
[Download the model](https://drive.google.com/file/d/1wAcHVZOVyXr5oFHy4eGr4YarGl6CsfC7/view?usp=sharing)

Please download this file and place it in the (`model/`) before running the application.
