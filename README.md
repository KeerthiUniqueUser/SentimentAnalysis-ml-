# Sentiment Analysis Machine Learning Project

## 📌 Project Overview

This project is an **end-to-end Machine Learning Sentiment Analysis application**. It trains an ML model to classify text reviews into **Positive** or **Negative** sentiments and provides a **Streamlit web interface** for user interaction. The project also covers **model training, evaluation, saving models, and deployment readiness**.

---

## 🎯 Objectives

* Build a complete **ML project** from scratch
* Perform **text preprocessing** and feature extraction
* Train and evaluate a **Machine Learning model**
* Create a **web app using Streamlit**
* Prepare the project for **deployment**

---

## 🛠️ Technologies Used

* **Python**
* **Pandas, NumPy** – Data handling
* **NLTK** – Text preprocessing
* **Scikit-learn** – ML model & evaluation
* **TF-IDF Vectorizer** – Feature extraction
* **Naive Bayes Classifier** – Sentiment classification
* **Streamlit** – Web application
* **Pickle** – Model serialization

---

## 📂 Project Structure

```
sentiment_analysis/
│── app.py                     # Streamlit web app
│── train_model.py              # Model training script
│── sentiment_model.pkl         # Trained ML model
│── vectorizer.pkl              # TF-IDF vectorizer
│── Train.csv                   # Training dataset
│── Valid.csv                   # Validation dataset
│── Test.csv                    # Test dataset
│── README.md                   # Project documentation
```

---

## 📊 Dataset Description

The project uses three datasets:

* **Train.csv** – Used to train the model
* **Valid.csv** – Used for validation
* **Test.csv** – Used for final testing

Each dataset contains:

* `text` → Review text
* `label` → Sentiment (0 = Negative, 1 = Positive)

---

## 🔄 Project Workflow

1. Load datasets
2. Clean and preprocess text data
3. Convert text to numerical features using **TF-IDF**
4. Train **Multinomial Naive Bayes** model
5. Validate and test the model
6. Save trained model and vectorizer
7. Build Streamlit UI for predictions

---

## 🧠 Model Details

* **Algorithm:** Multinomial Naive Bayes
* **Feature Extraction:** TF-IDF Vectorizer
* **Labels:**

  * `0` → Negative
  * `1` → Positive

---

## 📈 Evaluation Metrics

* Accuracy Score
* Classification Report (Precision, Recall, F1-score)

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```
pip install pandas numpy scikit-learn nltk streamlit
```

### 2️⃣ Train the Model

```
python train_model.py
```

This will generate:

* `sentiment_model.pkl`
* `vectorizer.pkl`

### 3️⃣ Run the Streamlit App

```
streamlit run app.py
```

---

## 🖥️ Web App Features

* User-friendly UI
* Real-time sentiment prediction
* Color-coded output (Green = Positive, Red = Negative)
* Image-based sentiment indication (optional)

---

