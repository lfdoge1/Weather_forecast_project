# Weather Forecast Project – LSTM-based Temperature Prediction

## 📌 Description

This project is a weather forecast system that uses LSTM (Long Short-Term Memory) deep learning models to predict temperature trends in **Delhi** and **Coventry**. It features:

- City-specific models trained on historical weather data
- Normalization with `joblib` scalers
- Server-client architecture using Python `socket`
- Natural language understanding for user queries (e.g., “What’s the weather in Coventry 3 days from now?”)

## 📁 Project Structure

```
weather-prediction/
├── client_socket.py               # Simple socket client to send messages
├── server_socket.py               # Server that loads models and replies with predictions
├── weather_Delhi.py               # Model training script for Delhi
├── weather_coventry.py           # Model training script for Coventry
├── data/                          # Original weather datasets
│   ├── weatherdata.csv           # Delhi raw data
│   └── coventry_data.csv         # Coventry raw data
├── outputs/                       # Trained models, scalers, and cleaned datasets (not uploaded)
│   ├── *.h5, *.pkl, *_clean.csv
├── .gitignore
└── README.md
```

## ⚙️ How to Run the Project

### 🔹 Step 1: Train the Models

Make sure your raw data is placed under the `data/` folder.

```bash
python weather_Delhi.py
python weather_coventry.py
```

These scripts will:
- Clean the data
- Train LSTM models
- Save models and scalers to `outputs/`

### 🔹 Step 2: Start the Server

```bash
python server_socket.py
```

This will:
- Load models and scalers
- Accept natural language input (like "weather in Delhi in 3 days")

### 🔹 Step 3: Start the Client

```bash
python client_socket.py
```

This allows you to type queries and receive temperature forecasts.

## 💬 Sample Query

```
> What is the weather in Delhi in 2 days?
< Forecast for Delhi on 2024-03-28 (2 day(s) from today): 28.62°C
```

## 📦 Dependencies

Install requirements with:

```bash
pip install -r requirements.txt
```

Make sure `tensorflow`, `joblib`, `pandas`, and `numpy` are included.

## 🙌 Author

**Xiang Li**  
MSc Telecommunications  
University College London