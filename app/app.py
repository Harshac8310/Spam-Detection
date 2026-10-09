import streamlit as st
import joblib

model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="🛡️",
    layout="centered"
)


st.title("Spam Message Detector")
st.caption("An NLP project powered by TF-IDF and Logistic Regression")
st.divider()

message = st.text_area(
    "Message to analyze",
    placeholder="Example: Congratulations! You've won a free prize...",
    height=150,
    help="Paste an SMS message to check whether it may be spam."
)
st.write("")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)[0]
        probabilities = model.predict_proba(message_tfidf)[0]

                
        if prediction == "spam":
            st.error("🚨 Spam Detected")
        else:
            st.success("✅ Likely a Legitimate Message")

        st.write("Ham probability")
        st.progress(float(probabilities[0]))

        st.write("Spam probability")
        st.progress(float(probabilities[1]))

