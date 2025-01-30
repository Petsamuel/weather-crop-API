import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Load JSON data
with open("crops.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Combine all crop categories into a single list
crop_data = []
for category in ["high_temp_low_humidity", "moderate_temp_high_humidity", "low_temp", "rainy", "default"]:
    if category in data:
        crop_data.extend(data[category])

# Convert to DataFrame
df = pd.DataFrame(crop_data)

# Ensure 'favorable_soil_type' exists
if "favorable_soil_type" in df.columns:
    df["favorable_soil_type"] = df["favorable_soil_type"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=["favorable_soil_type"], drop_first=True)

# Normalize numeric features (if applicable)
if "temp_range" in df.columns and "humidity_range" in df.columns:
    df[["temp_range", "humidity_range"]] = scaler.fit_transform(df[["temp_range", "humidity_range"]])

# Print the first few rows
print(df.head())
