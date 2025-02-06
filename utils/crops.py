from typing import Dict, List
import joblib
import json
import pandas as pd
import logging
from typing import Dict, Optional
from models import WeatherData, Coordinates
import numpy as np
from typing import List, Dict


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load crop dictionary from JSON file
with open('crop_dict.json', 'r') as f:
    CROP_DICT = {int(k): v for k, v in json.load(f).items()}

logging.info(f"Loaded crop dictionary: {CROP_DICT}")



# Load crop data from crops.json
with open("crops.json", "r") as f:
    crop_data = json.load(f)

# Load ecological zones data
with open('zones.json', 'r') as f:
    ECOLOGICAL_ZONES = json.load(f)
    
STATE_TO_ZONE = {}
for zone, data in ECOLOGICAL_ZONES.items():
    for state in data.get('States', []):        STATE_TO_ZONE[state.lower()] = zone

def get_soil_properties_by_location(state: str):
    """Get soil properties based on state location"""
    try:
        state = state.strip().title()
        
        if state.lower().endswith(' state'):
            state = state[:-6]  # Remove ' State' suffix
        zone = STATE_TO_ZONE.get(state.lower())
       
        
        if not zone:
            return None
            
        zone_data = ECOLOGICAL_ZONES[zone]
        return {
            'N': zone_data['N'],
            'P': zone_data['P'],
            'K': zone_data['K'],
            'ph': zone_data['pH'],
            'soil_type': zone_data['Soil'],
            'zone': zone
        }
    except Exception as e:
        logging.error(f"Error getting soil properties for {state}: {e}")
        return None

def extract_state_from_response(geocoding_data):
    """Extract and clean state name from geocoding response"""
    try:
        state = geocoding_data.get('state', '')
        
        if state.lower().endswith(' state'):
            state = state[:-6]
            
        state_mappings = {
            'fct': 'FCT',
            'federal capital territory': 'FCT',
            'abuja': 'FCT'
        }
        
        return state_mappings.get(state.lower(), state)
        
    except Exception as e:
        logger.error(f"Error extracting state: {e}")
        return None

def predict_crop(city: str, state: str, weather_data: dict, soil_props: dict):
    """Predict crop based on weather and soil data"""
    try:
        # Load model artifacts
        model = joblib.load('best_crop_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'N': soil_props['N'],
            'P': soil_props['P'],
            'K': soil_props['K'],
            'temperature': weather_data.get('temperature_2m', 25),
            'humidity': weather_data.get('humidity', 60),
            'ph': soil_props['ph'],
            'rainfall': weather_data.get('precipitation', 0)
        }])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        predictions = model.predict(input_scaled)
        
        # Convert predictions to list
        predictions = predictions.tolist()
        
        # Map predictions to crop names
        recommended_crops = [CROP_DICT.get(pred, "Unknown Crop") for pred in predictions]
        
        return {
            "status": "success",
            "location": {
                "city": city,
                "state": state,
                "ecological_zone": soil_props['zone']
            },
            "soil_properties": soil_props,
            "weather": weather_data,
            "recommended_crops": recommended_crops
        }
        
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def recommend_crops(weather_data, crop_data):
    temp = weather_data['temp']
    humidity = weather_data['humidity']
    main = weather_data['main']
    description = weather_data['description'].lower()
       
    if temp > 20 and humidity < 50:
        print("High temp, low humidity")
        return crop_data.get("high_temp_low_humidity", [])
    elif 20 >= temp > 15 and humidity >= 50:
        print("Moderate temp, high humidity")
        return crop_data.get("moderate_temp_high_humidity", [])
    elif temp < 15:
        print("Low temp")
        return crop_data.get("low_temp", [])
    elif 'rain' in description:
        print("Rainy")
        return crop_data.get("rainy", [])
    else:
        print("Default")
        return crop_data.get("default", [])