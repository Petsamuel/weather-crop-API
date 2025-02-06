# Crop Recommendation API

An API that recommends suitable crops based on weather conditions and soil properties.

## Features

- Weather data retrieval
- Soil property analysis
- Crop recommendations
- Historical weather data
- Forecast integration

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment variables
4. Run the API: `uvicorn main:app --reload`

## Environment Variables

Create a `.env` file with:

```env
WEATHER_API_KEY=your_key
WEATHER_API_URL=url
GEOCODING_API_URL=url
FORECAST_API_URL=url
WEATHER_API_KEY2=your_key
WEATHER_HISTORICAL_API_URL=url
CURRENT_AND_FORECAST_API_URL=url
CURRENT_IP_ADDRESS=ip

