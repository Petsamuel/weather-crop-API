import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load JSON data
with open("crops.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract crop data
crop_data = []
for key in ["high_temp_low_humidity", "moderate_temp_high_humidity", "low_temp", "rainy", "default"]:
    if key in data and isinstance(data[key], list):
        crop_data.extend(data[key])
    elif key in data and isinstance(data[key], dict):
        crop_data.append(data[key])

# Convert to DataFrame
df = pd.DataFrame(crop_data)
df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]

# Encode soil types
if "favorable_soil_type" in df.columns:
    df["favorable_soil_type"] = df["favorable_soil_type"].apply(
        lambda x: x.split(",") if isinstance(x, str) else [x] if isinstance(x, (str, list)) else []
    )
    mlb = MultiLabelBinarizer()
    encoded_soil = pd.DataFrame(mlb.fit_transform(df["favorable_soil_type"]), columns=mlb.classes_)
    df = df.drop(columns=["favorable_soil_type"]).join(encoded_soil)

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]
print("Columns in DataFrame:", df.columns.tolist())

# Features and Labels
y = df['crop']
soil_columns = [col for col in df.columns if col.lower() in ['loamy', 'sandy', 'arid', 'clay']]
X = df[soil_columns]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the model
joblib.dump(clf, "trained_model.pkl")

# Sample prediction
sample_input = pd.DataFrame([{'loamy': 1, 'sandy': 0, 'Arid': 0, 'clay': 0}], columns=X.columns)
predicted_crop = clf.predict(sample_input)
print("Predicted Crops:", predicted_crop)