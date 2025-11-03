# 📱 Spam SMS Detection using Machine Learning

This project is part of an internship task where we build an **AI model that classifies SMS messages as Spam or Legitimate (Ham)** using machine learning techniques such as **TF-IDF vectorization** and algorithms like **Naive Bayes**, **Logistic Regression**, and **Support Vector Machines (SVM)**.

The system can detect spam messages (like lottery, offers, or fake prizes) and classify normal texts accurately, with the trained model achieving over **98% accuracy**.

---

## 🚀 Project Overview

- **Goal:** Classify text messages as `SPAM` or `HAM` (legitimate).
- **Dataset:** Contains labeled SMS messages (`spam.csv`) with `v1` as label (`ham`/`spam`) and `v2` as message text.
- **Techniques Used:**
  - Text preprocessing (cleaning, stopword removal, punctuation removal)
  - Feature extraction using **TF-IDF (Term Frequency–Inverse Document Frequency)**
  - Model training using:
    - ✅ Multinomial Naive Bayes  
    - ✅ Logistic Regression  
    - ✅ Support Vector Machine (SVM)
  - Model evaluation using Accuracy, Precision, Recall, and F1-score
  - Saving and loading models using `joblib`
  - Interactive message testing with confidence scoring

---

## 💬 Key Features

- Clean text preprocessing (removes punctuation, numbers, and URLs)
- TF-IDF feature extraction
- Trains and compares 3 different ML models
- Automatically picks the best model
- Saves trained model for reuse
- Interactive testing with confidence score
- Warns user for borderline uncertain messages

---

## 🏗️ Project Structure

<img width="520" height="279" alt="image" src="https://github.com/user-attachments/assets/6de35b5f-e913-4f10-8f3e-d61216e4ae21" />

---

## 🧠 How It Works
🏋️ Train the Model
```bash
python spam_train.py
```
This will:
- Load and clean the dataset
- Split into training (80%) and testing (20%)
- Train Naive Bayes, Logistic Regression, and SVM
- Evaluate each model
- Select the best model based on F1 score
- Save it as (`models/best_model.joblib`)

Example Output
```bash
Training Model: Support Vector Machine
Accuracy: 0.9865
Precision: 0.9786
Recall: 0.9195
F1 Score: 0.9481

✅ Best Model: Support Vector Machine with F1 Score = 0.9481
💾 Model saved successfully to: models/best_model.joblib
```

---


## 📊 Model Performance Summary
| Model                             | Accuracy   | Precision  | Recall     | F1 Score   |
| --------------------------------- | ---------- | ---------- | ---------- | ---------- |
| Naive Bayes                       | 0.9650     | 0.9911     | 0.7450     | 0.8506     |
| Logistic Regression               | 0.9785     | 0.9252     | 0.9128     | 0.9189     |
| **Support Vector Machine (Best)** | **0.9865** | **0.9786** | **0.9195** | **0.9481** |

✅ Best Model: Support Vector Machine

✅ Saved at: models/best_model.joblib

---

## 🧪 Test the Model (Interactive Mode)
```bash
python spam_test.py
```
Then enter SMS messages interactively

Example Run:
```bash
Enter SMS message (or 'exit'): Congrats! You won a prize. Call 0800-123-456
Prediction: SPAM   | Confidence: 96.4%
✅ High confidence — model is very certain about this prediction.

Enter SMS message (or 'exit'): Hey, are we still meeting tomorrow?
Prediction: HAM    | Confidence: 97.9%
✅ High confidence — model is very certain about this prediction.

Enter SMS message (or 'exit'): You got selected from the form you filled last week. Come to the mall to collect the prize.
Prediction: HAM    | Confidence: 58.7%
⚠️ Low confidence — this message is borderline. Consider manual review.
```

## 🏁 Conclusion
This project successfully demonstrates a Spam SMS Detection System using traditional Machine Learning models.

By combining TF-IDF with Support Vector Machines, it achieves over 98% accuracy, showing that even lightweight ML pipelines can perform exceptionally well for text classification tasks.

