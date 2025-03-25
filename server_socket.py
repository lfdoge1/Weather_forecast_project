import socket
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np
import re
import os

output_dir = "outputs"

# ----------------------------
# Delhi Temperature Conversion
# ----------------------------
def fahrenheit_to_celsius(f_temp):
    return (f_temp - 32) * 5.0 / 9.0

# ----------------------------
# User input process
# ----------------------------

# Simple word-to-number map for small numbers (can expand)
WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12
}

def extract_offset_from_user_input(text):
    text = text.lower().strip()

    # 1. Common fixed phrases
    if re.search(r"\b(the\s+)?day\s+after\s+tomorrow\b", text):
        return 2
    elif re.search(r"\b(tomorrow|tomorrow’s|tomorrows)\b", text):
        return 1
    elif re.search(r"\b(next\s+week|upcoming\s+week|next\s+7\s+days|within\s+7\s+days)\b", text):
        return 7
    elif re.search(r"\bthis\s+weekend\b", text):
        return 2  # or adjust based on today’s weekday

    # 2. Digit-based expressions: in 3 days, after 4 days, 5 days later
    match = re.search(r"(?:in|after)?\s*(\d{1,2})\s*(day|days)\s*(later|from now)?", text)
    if match:
        return int(match.group(1))

    # 3. Word-number based expressions: in two days, five days later
    match_word = re.search(r"(?:in|after)?\s*(\w+)\s*(day|days)\s*(later|from now)?", text)
    if match_word:
        word = match_word.group(1)
        if word in WORD_NUMBERS:
            return WORD_NUMBERS[word]

    # Default: today
    return 0

# ----------------------------
# Socket Server Setup
# ----------------------------
server = socket.socket()
server.bind(("127.0.0.1", 8888))
server.listen(1)

print("\U0001F535 Chatbot server is starting... Waiting for connection.")
conn, addr = server.accept()
print(f"✅ Connected to client: {addr}")
print("Hello! How can I assist you with weather prediction today?")

# ----------------------------
# Load models & scalers per city
# ----------------------------
model_delhi = load_model(os.path.join(output_dir, "lstm_temperature_model_delhi.h5"), compile=False)
model_coventry = load_model(os.path.join(output_dir, "lstm_temperature_model_coventry.h5"), compile=False)

scalers_delhi = {
    "dew_point": joblib.load(os.path.join(output_dir, "dew_scaler_delhi.pkl")),
    "humidity": joblib.load(os.path.join(output_dir, "humidity_scaler_delhi.pkl")),
    "month": joblib.load(os.path.join(output_dir, "month_scaler_delhi.pkl")),
    "wind_speed": joblib.load(os.path.join(output_dir, "wind_scaler_delhi.pkl")),
    "year": joblib.load(os.path.join(output_dir, "year_scaler_delhi.pkl")),
    "temperature": joblib.load(os.path.join(output_dir, "temperature_scaler_delhi.pkl"))
}

scalers_coventry = {
    "dew_point": joblib.load(os.path.join(output_dir, "dew_scaler_coventry.pkl")),
    "humidity": joblib.load(os.path.join(output_dir, "humidity_scaler_coventry.pkl")),
    "month": joblib.load(os.path.join(output_dir, "month_scaler_coventry.pkl")),
    "wind_speed": joblib.load(os.path.join(output_dir, "wind_scaler_coventry.pkl")),
    "year": joblib.load(os.path.join(output_dir, "year_scaler_coventry.pkl")),
    "temperature": joblib.load(os.path.join(output_dir, "temperature_scaler_coventry.pkl"))
}

#print("[DEBUG] Models and scalers loaded.")

# ----------------------------
# Load & clean datasets
# ----------------------------
def clean_dataset(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

df_delhi = pd.read_csv(os.path.join(output_dir, "weather_delhi_clean.csv"))
df_coventry = pd.read_csv(os.path.join(output_dir, "weather_coventry_clean.csv"))
df_delhi["datetime"] = pd.to_datetime(df_delhi["datetime"])
df_delhi["month"] = df_delhi["datetime"].dt.month
df_delhi["year"] = df_delhi["datetime"].dt.year

df_coventry["datetime"] = pd.to_datetime(df_coventry["datetime"])
df_coventry["month"] = df_coventry["datetime"].dt.month
df_coventry["year"] = df_coventry["datetime"].dt.year

# ----------------------------
# Simulated "today" index
# ----------------------------
SIMULATED_TODAY_INDEX = 4988

# ----------------------------
# Main Chat Loop
# ----------------------------
while True:
    try:
        data = conn.recv(1024).decode()
        if not data:
            print(" Client disconnected.")
            break

        user_input = data.strip().lower()
        city = None


        if "delhi" in user_input:
            city = "Delhi"
            df = df_delhi
            model = model_delhi
            scalers = scalers_delhi
        elif "coventry" in user_input:
            city = "Coventry"
            df = df_coventry
            model = model_coventry
            scalers = scalers_coventry
        else:
            raise ValueError("Sorry, city not recognized. Please specify 'Delhi' or 'Coventry'.")

        offset = extract_offset_from_user_input(user_input)

        if offset > 7:
            raise ValueError("Only forecasts within 7 days are supported.")

        index = SIMULATED_TODAY_INDEX + offset
        if index < 3:
            raise ValueError("Not enough previous data to make a prediction.")
        if index >= len(df):
            raise ValueError(f"Prediction index {index} is out of range. Dataset has {len(df)} rows.")

        simulated_date = df.iloc[index]["datetime"] if "datetime" in df.columns else f"{offset} days later"

        # Prepare input features for the last 3 days
        features = ['dew_point', 'humidity', 'wind_speed', 'month', 'year']
        sequence = df[features].iloc[index - 3:index].copy()

        for col in features:
            sequence[col] = scalers[col].transform(sequence[[col]])

        input_seq = np.expand_dims(sequence.values, axis=0)  # Shape: (1, 3, 5)

        # Predict
        prediction = model.predict(input_seq)
        temperature = scalers["temperature"].inverse_transform(prediction)[0][0]
        if city.lower() == "delhi":
            temperature = fahrenheit_to_celsius(temperature)

        reply = (
            f"Forecast for {city} on {simulated_date} ({offset} day(s) from today): {temperature:.2f}°C"
        )

        print(f"✅ {user_input} → {reply}")

    except Exception as e:
        reply = f"⚠️ Error: {str(e)}"
        print("❌ Exception:", e)

    try:
        conn.send(reply.encode())
    except:
        print("⚠️ Unable to send message to client. Connection may have closed.")
        break

conn.close()