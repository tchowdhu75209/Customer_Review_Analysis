# config.py
import os

class Config:
    DATA_PATH = os.path.join(os.getcwd(), 'reviews_for_classification.csv')  # Construct absolute path
    MODEL_PATH = 'sentiment_model.pkl'
    TFIDF_PATH = 'tfidf_vectorizer.pkl'
    DEBUG = True  # Enable debug mode for Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'  # For session management