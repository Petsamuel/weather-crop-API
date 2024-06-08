from fastapi import FastAPI
from fastapi import HTTPException
import requests

app = FastAPI()
WEATHER_API_KEY = "your_api_key_here"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"


@app.get("/")
def read_root():
    return {"message": "Welcome to the Weather Crop API"}


@app.get("/weather")
def get_weather(city: str):
    params = {
        'q': city,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Weather data not found")
    return response.json()


def recommend_crops(weather_data):
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']

    if temp > 20 and humidity < 50:
        return ["Corn", "Wheat"]
    elif temp > 15 and temp <= 20 and humidity >= 50:
        return ["Rice", "Soybeans"]
    else:
        return ["Lettuce", "Carrots"]

@app.get("/recommendations")
def get_crop_recommendations(city: str):
    weather_data = get_weather(city)
    crops = recommend_crops(weather_data)
    return {"city": city, "crops": crops}
