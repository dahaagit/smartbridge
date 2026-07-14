from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import traceback
import feedparser
import urllib.parse

app = Flask(__name__)

# Load artifacts
try:
    artifacts = joblib.load('floods.save')
    model = artifacts['model']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    features = artifacts['features']
except Exception as e:
    print("Error loading model artifacts:", e)
    model, scaler, encoders, features = None, None, None, None

def fetch_news(query):
    # Use Google News RSS feed for live news
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
        news_items = []
        for entry in feed.entries[:4]: # Get top 4 news
            news_items.append({
                'title': entry.title,
                'link': entry.link,
                'published': entry.published
            })
        return news_items
    except Exception as e:
        print("Error fetching news:", e)
        return []

@app.route('/')
def home():
    # Fetch general flood news for India
    general_news = fetch_news("flood India")
    return render_template('home.html', news=general_news)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict_input')
def predict_input():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Please ensure 'floods.save' exists in the root directory.", 500

    try:
        # Extract features from form
        input_data = {}
        for feature in features:
            val = request.form.get(feature)
            if val is None:
                return f"Missing value for {feature}", 400
                
            if feature in encoders:
                try:
                    input_data[feature] = encoders[feature].transform([val])[0]
                except ValueError:
                    input_data[feature] = encoders[feature].transform([encoders[feature].classes_[0]])[0]
            else:
                input_data[feature] = float(val)

        # Get location text to fetch local news if provided, else general
        location = request.form.get('location_name', 'India')
        if not location.strip():
            location = "India"
        local_news = fetch_news(f"flood {location}")

        # Predict
        df_input = pd.DataFrame([input_data])
        X_scaled = scaler.transform(df_input)
        prediction = model.predict(X_scaled)[0]
        
        if prediction == 1:
            result_text = "Flood Chance Detected"
            status = "danger"
        else:
            result_text = "No Flood Chance"
            status = "safe"
            
        return render_template('result.html', result=result_text, status=status, location=location, news=local_news)

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
