# 🎬 Movie Genre Classification

A machine learning project that predicts the **genre of a movie** based on its **plot summary** using natural language processing (NLP) techniques.  
This project was developed and tested in **PyCharm**, using models trained with **TF-IDF** and **Logistic Regression / Naive Bayes**.

---

## 🧠 Project Overview

This project aims to **automatically classify movie genres** using textual plot summaries.  
It uses standard **text-based feature extraction** methods such as **TF-IDF Vectorization** and machine learning classifiers like:

- Naive Bayes  
- Logistic Regression  
- Support Vector Machines (SVM)

The model learns patterns in text to predict genres like *comedy, drama, thriller, horror, sci-fi, adventure,* and more.

---

## 📂 Dataset

The dataset used for training is a **custom movie dataset** containing the following fields:

| Column Name | Description |
|--------------|-------------|
| `id` | Unique movie ID |
| `title` | Movie title |
| `genre` | Movie genre (label) |
| `summary` | Plot summary / description text |

Example rows:

1 ::: Edgar's Lunch (1998) ::: thriller ::: L.R. Brane loves his life - his car, his apartment, his job, but especially his girlfriend...

2 ::: La guerra de papá (1977) ::: comedy ::: Spain, March 1964: Quico is a very naughty child of three belonging to a wealthy family...

---

##  Project Structure:

<img width="531" height="357" alt="image" src="https://github.com/user-attachments/assets/5953c7b1-ecc1-48c5-bcd1-69e908e16263" />

---

## ⚙️ Installation

### Clone this repository

```bash
git clone https://github.com/<your-username>/movie_genre_classification.git
cd movie_genre_classification

Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # For Windows
source .venv/bin/activate     # For macOS/Linux

Install dependencies
pip install -r requirements.txt
```
---

## Training the Model

To train the model using your dataset:
```bash
python train_genre_classifier.py
```
This will:

Load and preprocess the dataset

Vectorize the text using TF-IDF

Train a classification model

Save the trained files into the outputs/ directory


---

## Making Predictions (CLI Mode)

Use the trained model to predict genres from the terminal:
```bash
python predict_genre.py
```
You will see:

🎬 Movie Genre Predictor

Type a movie plot summary and press Enter.

Type 'exit' to quit.


Enter movie plot: A man wakes up to find his wife missing and a dark conspiracy unfolding.

🎯 Predicted Genre: mystery

## Making Predictions - Web App (Streamlit UI)
You can also run a web interface to classify movies interactively.

Run the app:
```bash
streamlit run web_app.py
```
Then open the link shown in the terminal (e.g., http://localhost:8501).

Features:

Simple text input box for plot summary.

“Predict” button to classify the genre

Example plot snippets for testing

Displays predicted genre instantly


## 🖼️ Results and Sample Outputs:
| Example Plot                                                                      | Predicted Genre |
| --------------------------------------------------------------------------------- | --------------- |
| “A group of children set out on a wild adventure to find a hidden treasure.”      | Adventure       |
| “A lonely astronaut drifts far from Earth and must struggle to survive.”          | Sci-Fi          |
| “A detective wakes up to find his wife missing and a dark government conspiracy.” | Mystery         |


---

## ⭐ Acknowledgements

Dataset inspired by open-source movie summary datasets

Built using Python, scikit-learn, pandas, and Streamlit


---
## Important Notice:
The dataset file zip is 45MB and too large to be uploaded to GitHub.
You can download the required dataset zip file from this link :- [dataset.zip](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb?resource=download)
Extract the zip file and place it inside the `data/` folder if you want to retrain the model
