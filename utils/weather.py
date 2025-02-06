from fastapi import HTTPException
from dotenv import load_dotenv
from models import WeatherData, Coordinates
import requests
import logging
import os

load_dotenv()

logger = logging.getLogger(__name__)

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")
GEOCODING_API_URL = os.getenv("GEOCODING_API_URL")
FORECAST_API_URL = os.getenv("FORECAST_API_URL")
WEATHER_API_KEY2 = os.getenv("WEATHER_API_KEY2")
WEATHER_HISTORICAL_API_URL = os.getenv("WEATHER_HISTORICAL_API_URL")
CURRENT_AND_FORECAST_API_URL = os.getenv("CURRENT_AND_FORECAST_API_URL")
CURRENT_IP_ADDRESS = os.getenv("CURRENT_IP_ADDRESS")

def get_weather(lat: float, lon: float):
    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    response = requests.get(WEATHER_API_URL, params=params)
   
    
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Weather data not found")
    data = response.json()
    
    return WeatherData(
        temp=data['main']['temp'],
        humidity=data['main']['humidity'],
        wind_speed=data['wind']['speed'],
        temp_min=data['main']['temp_min'],
        temp_max=data['main']['temp_max'],
        main=data['weather'][0]['main'],
        description=data['weather'][0]['description'],
    )

def current_weather(lat: float, lon: float):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "apparent_temperature", "is_day", "precipitation", "rain", "cloud_cover", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
	    "hourly": "temperature_2m",
        "timezone": "Africa/Cairo"
    }
    responses = requests.get(WEATHER_API_KEY2, params=params)
    if responses.status_code == 200:
        data = responses.json()
        # Extract specific variables from the current weather data
        current_data = data['current']
        return current_data
    else:
        # Handle the case where the API response is not successful
        return {"error": "Failed to retrieve weather data"}

def historical_weather(lat:float, lon:float, start_date: str, end_date: str):
    params = {
	"latitude": 52.52,
	"longitude": 13.41,
	"start_date": start_date,
	"end_date": end_date,
	"daily": ["temperature_2m_max", "temperature_2m_mean", "sunrise", "sunset", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours"],
	"timezone": "Africa/Cairo"
 }
    responses = requests.get(WEATHER_API_KEY2, params=params)
    if responses.status_code != 200:
        raise HTTPException(status_code=404, detail="Weather data not found")
    data = responses.json()
    return data

def get_coordinates(city: str):
    """Get coordinates and state for a city"""
    try:
        params = {
            'q': f"{city},NG",
            'limit': 1,
            'appid': WEATHER_API_KEY
        }
        
        response = requests.get(GEOCODING_API_URL, params=params)
        if response.status_code != 200 or len(response.json()) == 0:
            raise HTTPException(status_code=404, detail="City not found or not in Nigeria")
            
        data = response.json()[0]
        
        return data['lat'], data['lon'], data['state']
        
    except Exception as e:
        logger.error(f"Error getting coordinates for {city}: {e}")
        raise
# Get weather data for given coordinates

def get_weather_only(city: str):
    try:
        lat, lon, state = get_coordinates(city)
        weather_data = get_weather(lat, lon)
        
        return {
            "status": "success",
            "data": {
                "location": {
                    "city": city,
                    "state": state,
                    "coordinates": {"lat": lat, "lon": lon}
                },
                "weather": weather_data.dict()
            }
        }
        
    except HTTPException as e:
        return {"status": "error", "detail": e.detail}
    except ValueError as e:
        return {"status": "error", "detail": "Error processing coordinates"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    
def get_weather_forecast(lat: float, lon: float):
    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_API_KEY,
        'units': 'metric'
    }
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Weather data not found")
    data = response.json()
    return WeatherData(
        temp=data['main']['temp'],
        humidity=data['main']['humidity'],
        wind_speed=data['wind']['speed'],
        temp_min=data['main']['temp_min'],
        temp_max=data['main']['temp_max'],
        main=data['weather'][0]['main'],
        description=data['weather'][0]['main'],
    )