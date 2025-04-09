from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pickle
import folium
from geopy.geocoders import Nominatim
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords once
nltk.download('stopwords')

# Local preprocessing setup
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    return ' '.join(stemmer.stem(word) for word in words if word not in stop_words)

# Load model and vectorizer (only these two in model.pkl)
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# FastAPI app setup
app = FastAPI()

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TweetData(BaseModel):
    text: str
    location: str = None

class TweetRequest(BaseModel):
    tweets: List[TweetData]

# Geolocator
geolocator = Nominatim(user_agent="disaster_response_app")

@app.post("/generate-map")
async def classify_and_generate_map(request: TweetRequest):
    disaster_locations = []

    for tweet in request.tweets:
        try:
            processed_text = preprocess(tweet.text)
            vectorized = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized)[0]

            if prediction == 1 and tweet.location:
                location_data = geolocator.geocode(tweet.location)
                if location_data:
                    disaster_locations.append({
                        "text": tweet.text,
                        "location": tweet.location,
                        "lat": location_data.latitude,
                        "lon": location_data.longitude
                    })
        except Exception as e:
            print(f"Error with tweet '{tweet.text}': {e}")

    # Generate and return map
    folium_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    for loc in disaster_locations:
        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            popup=f"<strong>{loc['location']}</strong><br/>{loc['text']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(folium_map)

    return folium_map._repr_html_()
