# 📧 Spam Mail Detection using Machine Learning

A simple machine learning project that classifies SMS or email messages as **Spam** or **Not Spam (Ham)** using a **Multinomial Naive Bayes Classifier** with **TF-IDF vectorization**, deployed using **Streamlit**.

---

## 🚀 Live Demo

👉 [Click here to try the app on Streamlit]
(https://spamemaildetection-3ssx2cdjnam3e4h27ydrpe.streamlit.app/)



---

## 📂 Dataset

- **Dataset Name:** SMS Spam Collection Dataset  
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Format:** CSV  
- **Columns:**
  - `v1`: Label (ham or spam)
  - `v2`: Text message

---

## 🧠 Model & Features

- **Text Preprocessing** using `TF-IDF Vectorizer`
- **Classifier:** Multinomial Naive Bayes
- **Train/Test Split:** 80/20
- **Metrics:** Accuracy, Classification Report
- Real-time prediction with a user-friendly interface
- Displays model confidence and prediction results

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF
- Naive Bayes

---

## 📦 Folder Structure
.
├── spam.csv
├── streamlit_spam_detector.py
├── README.md

