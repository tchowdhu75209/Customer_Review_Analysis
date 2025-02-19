# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
from config import Config  # Import configuration

class SentimentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tfidf_vectorizer = None
        self.label_encoder = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.config.DATA_PATH)
            self.data = self.data[self.data['review_body'].notna()]
            self.data = self.data[self.data['review_head'].notna()]
            self.data['country'] = self.data['country'].fillna('US')
            return self.data
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.config.DATA_PATH}")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, data):
        from utils import clean_text
        data["cleaned_review_body"] = data["review_body"].apply(clean_text)
        return data

    def train_model(self, data):
        sentiment_map = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
        data["sentiment"] = data["stars"].map(sentiment_map)

        X_train, X_test, y_train, y_test = train_test_split(data["review_body"], data["sentiment"], test_size=0.2, random_state=42)

        self.label_encoder = LabelEncoder()
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)

        self.tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)

        self.model = MultinomialNB(alpha=0.5)
        self.model.fit(X_train_tfidf, y_train_enc)

        y_pred = self.model.predict(X_test_tfidf)
        print("Naive Bayes Performance:\n", classification_report(y_test_enc, y_pred, target_names=self.label_encoder.classes_))

        # Save the model and vectorizer
        pickle.dump(self.model, open(self.config.MODEL_PATH, 'wb'))
        pickle.dump(self.tfidf_vectorizer, open(self.config.TFIDF_PATH, 'wb'))
        print(f"Model and TF-IDF vectorizer saved to {self.config.MODEL_PATH} and {self.config.TFIDF_PATH}")

    def load_model(self):
        try:
            self.model = pickle.load(open(self.config.MODEL_PATH, 'rb'))
            self.tfidf_vectorizer = pickle.load(open(self.config.TFIDF_PATH, 'rb'))
            print(f"Model loaded from {self.config.MODEL_PATH} and TF-IDF vectorizer from {self.config.TFIDF_PATH}")
            return True
        except FileNotFoundError:
             print("Error: Model file not found. Please train the model first.")
             return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_sentiment(self, text):
        if self.model is None or self.tfidf_vectorizer is None:
            print("Model not loaded. Please load the model first.")
            return None

        cleaned_text = clean_text(text)
        text_tfidf = self.tfidf_vectorizer.transform([cleaned_text])
        predicted_label = self.model.predict(text_tfidf)[0]
        predicted_sentiment = self.label_encoder.inverse_transform([predicted_label])[0]
        return predicted_sentiment