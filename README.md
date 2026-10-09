# Spam Message Detector 📩

A machine learning project that classifies SMS messages as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP), TF-IDF vectorization, and Logistic Regression. The project includes an interactive web interface built with Streamlit.

## Features

- Classifies SMS messages as spam or legitimate.
- Converts text into numerical features using TF-IDF.
- Compares Multinomial Naive Bayes and Logistic Regression.
- Displays estimated spam and ham probabilities.
- Saves and reloads the trained model for predictions.
- Provides an interactive Streamlit web interface.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Logistic Regression
- Joblib
- Streamlit

## Dataset

This project uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

The dataset contains SMS messages labelled as:
- **Ham:** Legitimate messages.
- **Spam:** Unwanted or promotional messages.

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Project Workflow

1. Load and explore the SMS dataset.
2. Prepare message text and labels.
3. Split the dataset into training and testing sets.
4. Convert messages into numerical features using TF-IDF.
5. Train Multinomial Naive Bayes and Logistic Regression classifiers.
6. Evaluate and compare model performance.
7. Save the trained model and TF-IDF vectorizer.
8. Build an interactive application using Streamlit.

## Model Evaluation

The models were evaluated on a held-out test set containing 1,115 messages.

| Metric | Multinomial Naive Bayes | Logistic Regression |
|---|---:|---:|
| Accuracy | 96% | 97% |
| Spam Precision | 100% | 100% |
| Spam Recall | 70% | 80% |
| Spam F1-score | 0.83 | 0.89 |

Logistic Regression performed better on this test split, particularly in spam recall and F1-score. However, it still missed some spam messages, so the model should not be considered perfect.

*Note: Metrics are rounded. Results may vary with different data splits or training settings.*

## Project Structure

```text
Spam-Detector/
├── app/
│   └── app.py
├── data/
│   └── SMSSpamCollection
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── src/
│   ├── explore_data.py
│   └── predict.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Spam-Detector
```

Replace `<your-repository-url>` with your GitHub repository URL.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

From the project root directory, execute:

```bash
streamlit run app/app.py
```

The application will open in your browser. Enter an SMS message and click **Check Message** to see the prediction and estimated probabilities.

## Limitations

- Predictions depend on patterns learned from the training dataset.
- Informal, ambiguous, or unfamiliar messages may be misclassified.
- Estimated probabilities are not guarantees of correctness.
- The model may not generalize equally well to modern spam messages or messages in other languages.
- Evaluation results reflect one train-test split and may differ on other datasets.

## Future Improvements

- Evaluate the model using cross-validation.
- Improve spam recall while monitoring false positives.
- Test on newer and more diverse SMS messages.
- Add more comprehensive input validation and error handling.
- Deploy the application online.

## Learning Outcomes

This project provided practical experience with NLP preprocessing, TF-IDF feature extraction, supervised machine learning, model evaluation, model persistence, and building a web application with Streamlit.

## License

This project is intended for educational and learning purposes. Check the dataset's terms before redistributing it.