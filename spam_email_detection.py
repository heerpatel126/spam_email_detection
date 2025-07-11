import pandas as pd
import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Optionally: cache loading
@st.cache(show_spinner=False, allow_output_mutation=True)
def train_model():
    df = pd.read_csv('spam.csv', encoding='latin-1')[['v1','v2']]
    df.columns = ['label','message']
    df = df.drop_duplicates().dropna()
    df['label_num'] = df['label'].map({'ham':0,'spam':1})
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num'])
    vect = TfidfVectorizer(stop_words='english', min_df=2)
    X_train_vec = vect.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return vect, model

vect, model = train_model()

st.set_page_config(page_title="Spam Detector", layout="centered")
st.title("📧 SMS/Email Spam Detector")

user_input = st.text_area("Enter your message here:", height=150)

if st.button("Predict"):
    if user_input:
        vec = vect.transform([user_input])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][pred]
        label = "🛑 Spam" if pred == 1 else "✅ Not Spam"
        st.success(f"Prediction: **{label}** (confidence: {prob:.2%})")
    else:
        st.error("Please enter a message.")
