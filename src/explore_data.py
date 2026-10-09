import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import joblib


df = pd.read_csv("data/SMSSpamCollection",sep="\t",encoding="latin-1", header=None, names=["label", "message"])
#print(df.head())
#print(df.shape)
#print(df.columns)

#print(df["label"].value_counts())

X=df["message"]
y=df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#model = MultinomialNB()
#model.fit(X_train_tfidf, y_train)

#y_pred = model.predict(X_test_tfidf)

#accuracy = accuracy_score(y_test, y_pred)

#print("Model Accuracy:", accuracy)
#print("Model Accuracy (%):", accuracy * 100)
#print("\nClassification Report:")
#print(classification_report(y_test, y_pred))

#cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
#print("Confusion Matrix:")
#print(cm)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)

print("Logistic Regression Report:")
print(classification_report(y_test, lr_pred))

print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred, labels=["ham", "spam"]))


# Test with new, custom messages
custom_messages = [
    "Congratulations! You have won a free prize. Claim now!",
    "Hey, are we meeting at college tomorrow?",
    "URGENT! You have won cash. Call now to claim your reward!",
    "Hi, please call me when you reach home."
]

# Convert the messages using the existing TF-IDF vectorizer
custom_tfidf = vectorizer.transform(custom_messages)

# Predict using the trained Logistic Regression model
custom_predictions = lr_model.predict(custom_tfidf)
custom_probabilities = lr_model.predict_proba(custom_tfidf)

for message, prediction, probability in zip(custom_messages, custom_predictions, custom_probabilities):
    print("Message:", message)
    print("Prediction:", prediction)
    print("Probabilities:", probability)
    print("-" * 50)

joblib.dump(lr_model, "models/spam_classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
