
import joblib

# Load the saved model and vectorizer
model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Try a new message
message = ["Congratulations! You have won a free prize. Claim now!"]

# Convert text into TF-IDF features
message_tfidf = vectorizer.transform(message)

# Predict the label
prediction = model.predict(message_tfidf)

print("Message:", message[0])
print("Prediction:", prediction[0])
