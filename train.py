import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Load JSON data from the file
with open("crops.json", "r", encoding="utf-8") as file:
    data = json.load(file)  # Load JSON into a Python dictionary

# Initialize an empty list to store crop data
crop_data = []

# Iterate through different climate categories and extract crops
for key in ["high_temp_low_humidity", "moderate_temp_high_humidity", "low_temp", "rainy", "default"]:
    if key in data:  # Check if the key exists in JSON
        crop_data.extend(data[key])  # Append crop data to the list

# Convert extracted data into a Pandas DataFrame
df = pd.DataFrame(crop_data)

# Ensure the 'favorable_soil_type' column exists in the dataset
if "favorable_soil_type" in df.columns:
    # Convert soil type values into lists if they are not already
    df["favorable_soil_type"] = df["favorable_soil_type"].apply(lambda x: x if isinstance(x, list) else [x])

    # Use MultiLabelBinarizer to perform one-hot encoding on soil types
    mlb = MultiLabelBinarizer()
    encoded_soil = pd.DataFrame(mlb.fit_transform(df["favorable_soil_type"]), columns=mlb.classes_)

    # Merge the encoded data back into the DataFrame and remove the original column
    df = df.drop(columns=["favorable_soil_type"]).join(encoded_soil)

# Save the processed DataFrame as a pickle file for future use
joblib.dump(df, "crop_data.pkl")

# Save the processed DataFrame as a CSV file for easy inspection
df.to_csv("processed_crops.csv", index=False)

# Display the first few rows of the processed DataFrame
print(df.head())
