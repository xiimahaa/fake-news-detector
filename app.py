from flask import Flask, render_template, request
import joblib
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1iMahPaNtDGSXsW4Atd00dKs7rz6Aws8I"
VECTORIZER_URL = "https://drive.google.com/uc?id=1YTlVel8LcbAUYOyqYPCiEiu_kwfD7QS4"

if not os.path.exists("fake_news_rf_model.pkl"):
    gdown.download(MODEL_URL, "fake_news_rf_model.pkl", quiet=False)

if not os.path.exists("tfidf_vectorizer.pkl"):
    gdown.download(VECTORIZER_URL, "tfidf_vectorizer.pkl", quiet=False)
    
app = Flask(__name__)

model = joblib.load('fake_news_rf_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Text preprocessing
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization + stopword removal + lemmatization
    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(words)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    news = request.form['news']

    cleaned_news = clean_text(news)

    transformed_news = vectorizer.transform([cleaned_news])

    prediction = model.predict(transformed_news)[0]


    probability = model.predict_proba(transformed_news)

    confidence = round(max(probability[0]) * 100, 2)

    if prediction == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template(
        'index.html',
        prediction_text=result,
        confidence=confidence
    )

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
    
    
