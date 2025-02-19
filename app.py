# app.py (Flask application)
from flask import Flask, render_template, request, redirect, url_for
from model import SentimentAnalyzer
from config import Config

app = Flask(__name__)
app.config.from_object(Config)  # Load configuration from Config class

# Initialize the SentimentAnalyzer with the configuration
analyzer = SentimentAnalyzer(Config)

# Load the model at startup
with app.app_context():
    model_loaded = analyzer.load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review_text = request.form['review_text']
        if analyzer.model is None:
            return render_template('index.html', error="Model not loaded. Please train or load the model first.")

        sentiment = analyzer.predict_sentiment(review_text)
        return render_template('index.html', review_text=review_text, sentiment=sentiment)
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    data = analyzer.load_data()
    if data is None:
        return "Error loading data. Check the data path."
    processed_data = analyzer.preprocess_data(data)
    analyzer.train_model(processed_data)
    return "Model trained successfully! Please refresh the page."

@app.route('/load', methods=['GET'])
def load():
    if analyzer.load_model():
        return "Model loaded successfully! Please refresh the page."
    else:
        return "Model loading failed. Check the model path and ensure the model exists."

# Simple error handling for demonstration
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True) # Ensure debug=True is set in Config for production