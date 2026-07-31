import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download NLTK data
nltk.download('stopwords')

# Load Dataset
data = pd.read_csv("IMDB Dataset.csv")

print("First 5 Records:")
print(data.head())

# Stopwords and Stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Text Preprocessing Function
def preprocess(text):

    text = str(text).lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords + stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply preprocessing
data["clean_review"] = data["review"].apply(preprocess)

print("\nPreprocessed Data:")
print(data[["review", "clean_review"]].head())

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(data["clean_review"])
y = data["sentiment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model Training
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# User Input Prediction
user_review = input("\nEnter a movie review: ")

clean_review = preprocess(user_review)

review_vector = tfidf.transform([clean_review])

prediction = model.predict(review_vector)

print("\nPredicted Sentiment:", prediction[0])
