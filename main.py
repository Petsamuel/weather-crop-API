from fastapi import Depends, HTTPException, Header, FastAPI
import requests
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from models import WeatherData, Coordinates
from typing import List, Dict
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
import joblib
from sklearn.preprocessing import MinMaxScaler
import logging
import numpy as np
from utils.weather import get_weather, current_weather, historical_weather, get_coordinates,get_weather_forecast
from utils.crops import predict_crop, recommend_crops,get_soil_properties_by_location, extract_state_from_response
import time
from datetime import datetime
# Load environment variables from the .env file
load_dotenv()

# Get API Key from environment
API_KEY = os.getenv("API_SECRET_KEY")

# Dependency to verify API key
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crop Recommendation API NG",
    description="API for crop recommendations based on weather and soil data",
    version="0.0.1",
    docs_url="/",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],   
)
# Initialize cache immediately
FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")

# Load crop data from crops.json
with open("crops.json", "r") as f:
    crop_data = json.load(f)

# Load ecological zones data
with open('zones.json', 'r') as f:
    ECOLOGICAL_ZONES = json.load(f)

# Load crop dictionary from JSON file
with open('crop_dict.json', 'r') as f:
    CROP_DICT = json.load(f)

# Create state to zone mapping
STATE_TO_ZONE = {}
for zone, data in ECOLOGICAL_ZONES.items():
    for state in data.get('States', []):        STATE_TO_ZONE[state.lower()] = zone

@app.get("/secure", dependencies=[Depends(verify_api_key)])
def secure_data():
    return {"message": "You have access to the secured data!"}

@app.get("/current/weather/{city}", tags=["Weather"], status_code=200)
@cache( expire=300 )
def get_current_weather(city: str):
    try:
        lat, lon, state = get_coordinates(city)
        current_data = current_weather(lat, lon)
        states = extract_state_from_response(state)
        
        if 'error' in current_data:
            return {"status": "error", "detail": current_data['error']}
            
        return {
            "status": "success",
            "data": {
                "location": {
                    "city": city,
                    "state": states,
                    "coordinates": {"lat": lat, "lon": lon}
                },
                "weather": current_data
            }
        }
        
    except HTTPException as e:
        return {"status": "error", "detail": e.detail}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
 
@app.get("/weather/{city}", status_code=200,
    tags=["Weather"],
    summary="Get weather data for a city",
    description="Returns current weather data for the specified city"
         )
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
 
# TODO:crops to be plant in location
@app.get(
    "/cropToPlant/{crops}/{city}",
    status_code=200,
    summary="Get crops to plant in a location",
    description="Returns recommended crops to plant in a location",
    tags=["Crops"]
)
@cache(expire=300)
def get_crops_to_plant(crops: str, city: str):
    try:
        # Get coordinates and state
        coordinates = get_coordinates(city)
        if not coordinates or len(coordinates) != 3:
            return {"status": "error", "detail": f"Could not retrieve coordinates for {city}"}
        
        lat, lon, state = coordinates

        # Get soil properties
        soil_props = get_soil_properties_by_location(state)
        if not soil_props:
            logger.warning(f"No soil data found for {state}")
            return {"status": "error", "detail": f"No soil data found for {state}"}

        # Get recommended crops for this zone
        zone = soil_props['zone']
        recommended_crops = ECOLOGICAL_ZONES.get(zone, {}).get("Crops", [])

        # Process user input crops
        crop_list = [crop.strip().lower() for crop in crops.split(',') if crop.strip()]
        
        # Identify suitable and unsuitable crops
        suitable_crops = [crop for crop in crop_list if crop in recommended_crops]
        unsuitable_crops = [crop for crop in crop_list if crop not in recommended_crops]

        # Log extracted data
        logger.info(f"Fetching crops for city: {city}, state: {state}, zone: {zone}")
        logger.info(f"User entered crops: {crop_list}")
        logger.info(f"Suitable crops: {suitable_crops}")
        logger.info(f"Unsuitable crops: {unsuitable_crops}")

        # Prepare response
        response = {
            "status": "success",
            "location": {
                "city": city,
                "state": state,
                "zone": zone,
                "coordinates": {"lat": lat, "lon": lon}
            },
            "soil_properties": soil_props,
            "recommended_crops": recommended_crops,
            "user_crops": crop_list,
            "suitable_crops": suitable_crops,
        }

        # If some crops are unsuitable, return a warning
        if unsuitable_crops:
            response["unsuitable_crops"] = unsuitable_crops
            response["message"] = (
                f"The following crops may not be ideal for {city}: {', '.join(unsuitable_crops)}. "
                f"Recommended crops for this area: {', '.join(recommended_crops)}."
            )

        return response

    except HTTPException as e:
        logger.error(f"HTTP error in get_crops_to_plant: {e.detail}")
        return {"status": "error", "detail": e.detail}

    except Exception as e:
        logger.exception(f"Unexpected error in get_crops_to_plant: {e}")
        return {"status": "error", "detail": str(e)}
 

# Get weather and recommendations for a city
@app.get("/recommend-crops/{city}",
    status_code=200,
    tags=["Recommendations"],
    summary="Get crop recommendations",
    description="Returns recommended crops based on location and conditions")
@cache( expire=300,dependencies=[Depends(verify_api_key)] )
async def recommend_crops(city: str):
    try:
        # Get location and weather data
        lat, lon, state = get_coordinates(city)
        weather_data = current_weather(lat, lon)
        
        if 'error' in weather_data:
            return {"status": "error", "detail": weather_data['error']}
        
        # Get soil properties
        soil_props = get_soil_properties_by_location(state)
        if not soil_props:
            return {"status": "error", "detail": f"No soil data found for {state}"}
            
        # Get prediction
        return predict_crop(city, state, weather_data, soil_props)
        
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    
# api for forecasting
@app.get("/weather/forecast/{city}", status_code=200, tags=["Forecast"], dependencies=[Depends(verify_api_key)])
def get_weather_forecast_and_crop_recommendations(city: str):
    try:
        lat, lon = get_coordinates(city)
        weather_data = get_weather_forecast(lat, lon)
        # recommended_crops = recommend_crops(weather_data.dict(), crop_data)
        return {"status": "success", "data": weather_data.dict()}
    except HTTPException as e:
        return {"status": "error", "detail": e.detail}


@app.get("/weather/history/{city}/{start_date}/{end_date}", status_code=200, summary="start_date and end_date: 2022-01-01, 2022-01-31", tags=["History"])
def historical_weather_data(city: str, start_date: str, end_date: str):
    #examples of start_date and end_date: 2022-01-01, 2022-01-31
    lat, lon, state = get_coordinates(city)
    historical_data = historical_weather(lat, lon, start_date, end_date)
    return {"status": "success", "data": historical_data}

#health
@app.get("/health", status_code=200, summary="Check the health of the API", tags=["Health"],)
@cache( expire=300 )
def health():
    start_time = time.time()
    response_time = round(time.time() - start_time, 4)
    # Returns a message describing the status of this service
    return {
        "status":"success",
        "message":"The service is running",
      "timestamp": datetime.utcnow().isoformat(),
         "response_time": response_time,
        "version": "0.0.1"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)

