from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from fastapi.responses import HTMLResponse  # ✅ Add this
import pickle
import folium
from geopy.geocoders import Nominatim
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Define preprocess function (since we didn't pickle it)
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    return ' '.join(stemmer.stem(w) for w in words if w not in stop_words)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

# FastAPI setup
app = FastAPI()

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

# Geolocation
geolocator = Nominatim(user_agent="disaster_response_app")

# ✅ Fix: use HTMLResponse and full map HTML
@app.post("/generate-map", response_class=HTMLResponse)
async def classify_and_generate_map(request: TweetRequest):
    disaster_locations = []

    for tweet in request.tweets:
        try:
            processed_text = preprocess(tweet.text)
            vectorized = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized)[0]

            if prediction == 1 and tweet.location:
                loc_data = geolocator.geocode(tweet.location)
                if loc_data:
                    disaster_locations.append({
                        "text": tweet.text,
                        "location": tweet.location,
                        "lat": loc_data.latitude,
                        "lon": loc_data.longitude
                    })
        except Exception as e:
            print(f"Error for tweet '{tweet.text}': {e}")

    # Create map
    folium_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    for loc in disaster_locations:
        folium.Marker(
            location=[loc["lat"], loc["lon"]],
            popup=f"<strong>{loc['location']}</strong><br/>{loc['text']}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(folium_map)

    # ✅ Return full HTML string (browser-friendly)
    return folium_map.get_root().render()
