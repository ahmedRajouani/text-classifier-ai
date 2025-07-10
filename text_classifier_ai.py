# file: text_classifier_ai.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string
import re

# Download NLTK stopwords
nltk.download("stopwords")

# 1. Sample dataset (replace with CSV later)
data = pd.DataFrame({
    'text': [
        "Buy cheap products now!",
        "Meeting scheduled at 10am tomorrow",
        "Limited time offer, act now!",
        "Can we reschedule our call?",
        "Congratulations, you won a free ticket!",
        "Let's review the report later",
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
})

# 2. Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

data['clean_text'] = data['text'].apply(clean_text)

# 3. Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 5. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# 7. Predict new message
def classify_message(msg):
    cleaned = clean_text(msg)
    vect = vectorizer.transform([cleaned])
    return model.predict(vect)[0]

# Example:
new_msg = "Get your free vacation now!"
print("New message:", new_msg)
print("Prediction:", classify_message(new_msg))
