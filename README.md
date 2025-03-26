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
├── weather_coventry.py            # Model training script for Coventry
├── data/                          # Original weather datasets
│   ├── weatherdata.csv            # Delhi raw data
│   └── coventry_data.csv          # Coventry raw data
├── outputs/                       # Trained models, scalers, and cleaned datasets (not uploaded)
│   ├── *.h5, *.pkl, *_clean.csv
├── requirements.txt               # Auto-generated via pip freeze
├── .gitignore
└── README.md
```

## ⚙️ How to Run the Project

### 🔹 Step 0: Python 3.12

### 🔹 Step 1: Clone the Repository

```bash
git clone https://github.com/lfdoge1/Weather_forecast_project.git
cd Weather_forecast_project
```

### 🔹 Step 2: (Optional) Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# or
source venv/bin/activate     # On macOS/Linux
```

### 🔹 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 4: Train Models

```bash
python weather_Delhi.py
python weather_coventry.py
```
        ⚠️ ⚠️ ⚠️ Warning: You must run both main programs, otherwise the server and client will not work.
        
        ⚠️ ⚠️ ⚠️ NOTE: The main program will not terminate until all plot windows are closed.

This will:
- Clean the data
- Train LSTM models
- Save models and scalers into `outputs/`

### 🔹 Step 5: Start the Server

```bash
python server_socket.py
```

### 🔹 Step 6: Start the Client

```bash
python client_socket.py
```

Enter natural language queries like:

```
> What is the weather in Delhi in 2 days?
< Forecast for Delhi on 2025-03-28 (2 day(s) from today): 28.62°C
```

---

## 📦 Dependencies

All required libraries are listed in `requirements.txt`, auto-generated via:

```bash
pip freeze > requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🙌 Author

**Xiang Li**  
MSc Telecommunications  
University College London
