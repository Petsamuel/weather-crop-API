from pydantic import BaseModel

class Coordinates(BaseModel):
    lat: float
    lon: float

class WeatherData(BaseModel):
    temp: float
    humidity: int
    wind_speed: float
    description: str