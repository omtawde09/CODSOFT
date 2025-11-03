# 🧠 Customer Churn Prediction

## 📋 Project Overview
This project aims to **predict customer churn** — identifying customers likely to discontinue using a service — using **Machine Learning** techniques.  
The goal is to help businesses take proactive actions to improve customer retention.

The model is trained on the **Churn Modelling dataset**, containing historical data about bank customers, including demographics, account information, and activity details.  
Multiple models were trained and compared to identify the best-performing one.

---

## 📁 Project Structure

<img width="710" height="357" alt="image" src="https://github.com/user-attachments/assets/2a79f5dc-67f1-4f5b-a936-f9774eec818f" />

---

## 🧾 Dataset Description
**Dataset Name:** `Churn_Modelling.csv`  
**Total Records:** 10,000  
**Total Features:** 14  
**Target Variable:** `Exited`  
- `1` → Customer has churned  
- `0` → Customer is retained

### 📊 Feature Summary

| Feature | Description |
|----------|--------------|
| CreditScore | Customer's credit score |
| Geography | Country (France / Spain / Germany) |
| Gender | Male / Female |
| Age | Age of the customer |
| Tenure | Number of years the customer has been with the bank |
| Balance | Account balance |
| NumOfProducts | Number of products used by the customer |
| HasCrCard | 1 if customer has a credit card, 0 otherwise |
| IsActiveMember | 1 if customer is active, 0 otherwise |
| EstimatedSalary | Estimated annual salary |
| Exited | Target variable (1 = churned, 0 = stayed) |

Irrelevant columns like `RowNumber`, `CustomerId`, and `Surname` were removed during preprocessing.

---

## ⚙️ Project Workflow

### 🧩 1. Data Loading and EDA
- Loaded dataset using **pandas**.
- Checked data types, missing values, and basic statistics.
- Verified class imbalance (≈20% churn rate).

### ⚙️ 2. Data Preprocessing
- Dropped unnecessary columns.
- Scaled numerical features using **StandardScaler**.
- Encoded categorical features using **OneHotEncoder**.
- Split data into **80% training** and **20% testing**.

### 🤖 3. Model Training
Trained and compared the following algorithms:
- **Logistic Regression**
- **Random Forest Classifier**
- **XGBoost Classifier**

Each model was evaluated using multiple metrics to determine performance.

### 📈 4. Model Evaluation
Evaluated each model using:
- **Accuracy**
- **F1 Score**
- **ROC-AUC Score**
- **Classification Report**
- **Confusion Matrix**

### 💾 5. Model Saving
The best-performing model, along with its **preprocessor** and **scaler**, was saved using `joblib` into the `models/` directory for future predictions.

### 💬 6. Interactive CLI Prediction
A command-line script (`predict_churn.py`) allows users to input customer details (like Age, Balance, etc.) and receive churn predictions with probability scores.

---

## 🧠 Model Performance Summary

| Model | Accuracy | F1-Score | ROC-AUC |
|--------|-----------|----------|----------|
| Logistic Regression | 0.71 | 0.50 | 0.77 |
| Random Forest | **0.86** | 0.56 | **0.85** |
| XGBoost | 0.84 | 0.55 | 0.82 |

✅ **Best Model:** Random Forest Classifier  
⭐ **Reason:** Highest ROC-AUC (0.85) and accuracy (86%)

---

## 💻 How to Run This Project

### 🪄 1. Clone the Repository
```bash
git clone https://github.com/omtawde09/CODSOFT/customer-churn-prediction.git
cd customer-churn-prediction
```
## 🧰 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/Scripts/activate      # On Windows
# or
source .venv/bin/activate          # On macOS/Linux
```
## 📦 3. Install Dependencies
```bash
pip install -r requirements.txt
```
## 4. 🧠Train our own model or use the pre-trained model
To train new model make sure the dataset (`Churn_Modelling.csv`) is placed inside the (`data/`) folder.
```bash
python customer_churn_prediction.py
```
## 🔮 5. Make Predictions (Interactive CLI)
Run the prediction script:

```bash
python predict_churn.py
```

You will be prompted to enter customer details

Example:
```bash
Credit Score (e.g. 600): 700
Geography (France / Spain / Germany): France
Gender (Male / Female): Male
Age: 35
Tenure (years with bank): 5
Account Balance: 60000
Number of Products (1-4): 2
Has Credit Card (1 = Yes, 0 = No): 1
Is Active Member (1 = Yes, 0 = No): 1
Estimated Salary: 90000
```

## Important Notice: Model File

The trained model file (`best_model.joblib`) is 40MB and too large to be uploaded to GitHub.

**You can download the required `best_model.joblib` file from this link:**
[Download the model](https://drive.google.com/file/d/1p3dn5hLIuBdKUk8c_DePW2iKPCNIhVoF/view?usp=sharing)

Please download this file and place it in the (`model/`) before running the application.

---
