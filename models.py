from pydantic import BaseModel
from typing import Optional, List
class Coordinates(BaseModel):
    lat: float
    lon: float

class WeatherData(BaseModel):
    temp: float
    humidity: int
    wind_speed: float
    temp_min:float
    temp_max:float
    main:str
    description: str

class WeatherData(BaseModel):
    temp: float
    humidity: float
    wind_speed: float
    temp_min: float
    temp_max: float
    main: str
    description: str

class CropRecommendation(BaseModel):
    crop_name: str
    confidence_score: float
    growing_season: Optional[str]
    water_requirements: Optional[str]
    soil_requirements: List[str]

class WeatherResponse(BaseModel):
    status: str
    weather_data: WeatherData
    crop_recommendations: List[CropRecommendation]