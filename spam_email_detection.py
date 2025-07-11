import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

@st.cache_resource(show_spinner=False)
def train_model():
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["message"]
    y = df["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return vectorizer, model, accuracy, report

# Load model + vectorizer
vectorizer, model, accuracy, report = train_model()

# Streamlit UI
st.set_page_config(page_title="Spam Detector App", layout="centered")
st.title("📧 Spam Message Detector")
st.write(f"### ✅ Model Accuracy: {accuracy:.2%}")
st.text(report)

user_input = st.text_area("Enter a message:", height=150)

if st.button("Predict"):
    if user_input:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        label = "🛑 Spam" if prediction == 1 else "✅ Not Spam"
        st.success(f"Prediction: **{label}**")
    else:
        st.warning("Please enter a message.")
