# Fake News Detection using Natural Language Processing (NLP)

A Machine Learning and Natural Language Processing (NLP) project for detecting whether a news article is **Real** or **Fake** using multiple classification algorithms and text preprocessing techniques.

This project was developed as part of a practical machine learning research project focused on transforming ML experiments into research-paper style implementations.

---

## 📌 Project Overview

Fake news has become a major challenge in the digital era due to the rapid spread of misinformation across online platforms and social media. This project aims to automatically classify news articles as either **Fake** or **Real** using NLP preprocessing and machine learning classification models.

The system processes textual news data, extracts meaningful features using TF-IDF vectorization, and trains multiple machine learning models to compare their performance.

The final deployed model is based on the **Random Forest Classifier**, which achieved the highest performance among all tested models.

---

## 📊 Dataset

This project uses the **Fake and Real News Dataset** from Kaggle:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

### Dataset Information

The dataset contains two CSV files:

- `Fake.csv` → fake news articles  
- `True.csv` → real news articles  

### Dataset Features

- **Title** → title of the news article  
- **Text** → full news article content  
- **Subject** → article category  
- **Date** → publication date  

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Matplotlib  
- Seaborn  
- Joblib  
- Flask  

---

## 🧠 NLP Techniques Used

- Lowercasing text  
- Removing URLs  
- Removing punctuation  
- Removing stopwords  
- Lemmatization  
- TF-IDF Vectorization  

---

## 🤖 Machine Learning Models

- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest Classifier  

---

## 📈 Model Performance

| Model | Accuracy |
|------|----------|
| Random Forest | 99.75% |
| SVM | 99.51% |
| Logistic Regression | 98.84% |
| Naive Bayes | 94.55% |

Based on evaluation metrics:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC Curve & AUC Score  

The **Random Forest Classifier** was selected as the final model.

---

## 🌐 Web Application

The project includes a Flask-based web application where users can:

- Enter a news article  
- Click "Analyze"  
- Get prediction:
  - Fake News ❌  
  - Real News ✅  
- View confidence score  

---

## 📁 Project Structure
FakeNewsDetectionApp/  
│  
├── app.py  
├── fake_news_rf_model.pkl   (larger than 25MB – cannot be uploaded to GitHub)  
├── tfidf_vectorizer.pkl     (larger than 25MB – cannot be uploaded to GitHub)  
├── requirements.txt  
├── Procfile  
│  
├── templates/  
│   └── index.html  
│  
├── static/  
│   └── style.css  
│  
└── ARTI406_Fake_News_Detectionn.ipynb



