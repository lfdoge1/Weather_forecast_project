import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os

import plotly
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf
from IPython.display import display

import warnings

warnings.filterwarnings("ignore")
warnings.warn("this will not show")

plt.rcParams["figure.figsize"] = (10, 6)

sns.set_style("whitegrid")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Set it None to display all rows in the dataframe
# pd.set_option('display.max_rows', None)

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)

# ----------------------------
# Read Data
# ----------------------------

data_dir = "data"
output_dir = "outputs"

os.makedirs(output_dir, exist_ok=True)

raw_data_path = os.path.join(data_dir, "weatherdata.csv")
df = pd.read_csv(raw_data_path)

# Rename 'Date' to 'day'
df = df.rename(columns={'Date': 'day'})

# Transform to  datetime
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'day']])
df = df.set_index('datetime')

# Rename
df = df.rename(columns={
    'Avg Dew Point': 'dew_point',
    'Avg Humidity': 'humidity',
    'Avg Wind Speed': 'wind_speed',
    'Avg Pressure': 'pressure',
    'Total Precipitation': 'precipitation',
    'Avg Temperature': 'temperature'
})

df = df[['dew_point', 'humidity', 'wind_speed', 'pressure', 'precipitation', 'temperature']]
clean_data_path = os.path.join(output_dir, "weather_delhi_clean.csv")
df.to_csv(clean_data_path)

df.info()

df.describe().T

# creating new features for EDA

df["year"] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df["date"] = df.index.date

df.head()
df.info()

#================
#     EDA
#================

# creating dark gray template(xgridoff_dark) from xgridoff template:

import plotly.io as pio

# Get the 'xgridoff' template
xgridoff_template = pio.templates['xgridoff']

# Customize the template for dark mode
xgridoff_template.layout.paper_bgcolor = 'rgb(25, 25, 25)'  # very dark gray background
xgridoff_template.layout.plot_bgcolor = 'rgb(35, 35, 35)'  # Dark gray plot background

xgridoff_template.layout.font.color = 'lightgray'  # Light gray font color

# Adjust gridline color and width
xgridoff_template.layout.xaxis.gridcolor = 'rgba(200, 200, 200, 0.3)'
xgridoff_template.layout.yaxis.gridcolor = 'rgba(200, 200, 200, 0.3)'
xgridoff_template.layout.xaxis.gridwidth = 1
xgridoff_template.layout.yaxis.gridwidth = 1

# Update Plotly templates with the modified 'xgridoff' template
pio.templates['xgridoff_dark'] = xgridoff_template

# =====================================
#  Plotting features over time (daily)
# =====================================

columns = {
    "temperature": "Temperature Over Time",
    "humidity": "Humidity Over Time",
    "wind_speed": "Wind Speed Over Time",
    "pressure": "Pressure Over Time",
    "dew_point": "Dew Point Over Time",
    "precipitation": "Precipitation Over Time"
}

for col, title in columns.items():
    fig = px.line(df, x=df.index, y=col, title=title)
    fig.update_layout(template='xgridoff_dark', title_x=0.5, xaxis_title="Date")
    fig.show()


# ===========================================
# Plotting Seasonal Decompositions with plotly
# ===========================================

from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.subplots as sp

# Perform seasonal decomposition
result = seasonal_decompose(df['temperature'], model='additive', period=365)

# Plot the decomposed components
fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True,
                       subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])

fig.add_trace(go.Scatter(x=df.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=result.resid, mode='lines', name='Residual'), row=4, col=1)

fig.update_layout(template='xgridoff_dark', height=800, title='Seasonal Decomposition of Mean Temperature')
fig.show()

# ===========================================
# Alternative Seasonal Decomposition plot
# ===========================================

import statsmodels.api as sm

plt.rcParams['figure.figsize'] = [15, 7]

# Select the 'meantemp' column and resample it to monthly frequency
data_monthly = df['temperature'].resample('M').mean()

# Perform seasonal decomposition for 'meantemp' feature
decomposition = sm.tsa.seasonal_decompose(data_monthly)

# Plot the decomposition
fig = decomposition.plot()
plt.show()



import plotly.graph_objects as go
features = ['temperature', 'pressure', 'humidity', 'wind_speed', 'dew_point', 'precipitation']
titles = [
    'Monthly Avg Temperature',
    'Monthly Avg Pressure',
    'Monthly Avg Humidity',
    'Monthly Avg Wind Speed',
    'Monthly Avg Dew Point',
    'Monthly Total Precipitation'
]

# ===========================================
#  Creat heatmap
# ===========================================
def create_heatmap_trace(data, feature):
    heatmap_data = data.pivot_table(values=feature, index='year', columns='month', aggfunc='mean')
    heatmap_text = heatmap_data.round(2).astype(str).values

    heatmap = go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='thermal',
        text=heatmap_text,
        hoverinfo='text'
    )

    annotations = [
        go.layout.Annotation(
            x=heatmap_data.columns[j],
            y=heatmap_data.index[i],
            text=heatmap_text[i][j],
            showarrow=False,
            font=dict(
                color='white' if heatmap_data.values[i, j] < (heatmap_data.values.max() / 2) else 'black'
            )
        )
        for i in range(len(heatmap_data.index))
        for j in range(len(heatmap_data.columns))
    ]

    return heatmap, annotations

#  trace
heatmap_traces = []
annotations_list = []

for feature in features:
    trace, annotations = create_heatmap_trace(df, feature)
    heatmap_traces.append(trace)
    annotations_list.append(annotations)

# creat figure
fig = go.Figure(data=heatmap_traces)
for i, trace in enumerate(fig.data):
    trace.visible = (i == 0)

# drop down button
dropdown_buttons = [
    dict(
        args=[
            {'visible': [j == i for j in range(len(features))]},
            {
                'annotations': annotations_list[i],
                'title': titles[i]
            }
        ],
        label=titles[i],
        method='update'
    )
    for i in range(len(features))
]

fig.update_layout(
    title=titles[0],
    xaxis=dict(title='Month'),
    yaxis=dict(title='Year'),
    annotations=annotations_list[0],
    width=1000,
    height=600,
    font=dict(size=10),
    margin=dict(l=60, r=60, t=80, b=60),
    updatemenus=[
        dict(
            buttons=dropdown_buttons,
            direction='down',
            showactive=True,
            x=1.15,
            y=1.15
        )
    ]
)

fig.show()

# ===========================================
#  Heatmap with seaborn
# ===========================================


plt.figure(figsize=(13, 6))

sns.heatmap(df.select_dtypes('number').corr(), cmap='Blues', annot=True, fmt='.2f');

# ===========================================
#  Correlation Bar plot with meantemp feature
# ===========================================

plt.figure(figsize=(15, 8))
sns.set_style('darkgrid')

correlation_matrix = round(df.select_dtypes('number').corr(), 2)

correlation_with_trgt = correlation_matrix['temperature'].sort_values(ascending=False)

ax = sns.barplot(x=correlation_with_trgt.index, y=correlation_with_trgt, palette='viridis')

plt.title('Correlation with temperature', size=20)
plt.xlabel('Features')
plt.ylabel('Correlation')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.xticks(rotation=45, ha='right')
plt.show()

# ===========================================
# Box plot
# ===========================================

# Box plots by Month

for feature in features:
    fig = px.box(df, x='month', y=feature,
                 title=f'Monthly Distribution of {feature.replace("_", " ").title()}',
                 template='xgridoff_dark')
    fig.show()

# Feature Selection
df = df[['temperature', 'dew_point', 'humidity', 'wind_speed', 'month', 'year']]
df.head()

df.info()

# Creating a new Dataframe for ARIMA-SARIMA Models
df1 = df.copy()
df1

#==========================
# Data Preprocessing
#==========================

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
dl_train, dl_test = df.iloc[:train_size], df.iloc[train_size:]
print(len(dl_train), len(dl_test))

# find out what scaling to perform

features = ['dew_point', 'humidity', 'wind_speed', 'month','year']

plt.figure(figsize=(16, 4))
sns.set_style('darkgrid')

# Applying:
# Robust scaling for all

from sklearn.preprocessing import RobustScaler, MinMaxScaler
import joblib

#==========================
# scaler
#==========================

dew_scaler = RobustScaler()
humidity_scaler = RobustScaler()
wind_scaler = RobustScaler()
month_scaler = MinMaxScaler()
year_scaler = MinMaxScaler()
target_transformer = MinMaxScaler()

dl_train['dew_point'] = dew_scaler.fit_transform(dl_train[['dew_point']])
dl_train['humidity'] = humidity_scaler.fit_transform(dl_train[['humidity']])
dl_train['wind_speed'] = wind_scaler.fit_transform(dl_train[['wind_speed']])
dl_train['month'] = month_scaler.fit_transform(dl_train[['month']])
dl_train['year'] = year_scaler.fit_transform(dl_train[['year']])
dl_train['temperature'] = target_transformer.fit_transform(dl_train[['temperature']])

# ===== test just transform =====
dl_test['dew_point'] = dew_scaler.transform(dl_test[['dew_point']])
dl_test['humidity'] = humidity_scaler.transform(dl_test[['humidity']])
dl_test['wind_speed'] = wind_scaler.transform(dl_test[['wind_speed']])
dl_test['month'] = month_scaler.transform(dl_test[['month']])
dl_test['year'] = year_scaler.transform(dl_test[['year']])
dl_test['temperature'] = target_transformer.transform(dl_test[['temperature']])

#====== save transform =====================
joblib.dump(dew_scaler, os.path.join(output_dir, "dew_scaler_delhi.pkl"))
joblib.dump(humidity_scaler, os.path.join(output_dir, "humidity_scaler_delhi.pkl"))
joblib.dump(wind_scaler, os.path.join(output_dir, "wind_scaler_delhi.pkl"))
joblib.dump(month_scaler, os.path.join(output_dir, "month_scaler_delhi.pkl"))
joblib.dump(year_scaler, os.path.join(output_dir, "year_scaler_delhi.pkl"))
joblib.dump(target_transformer, os.path.join(output_dir, "temperature_scaler_delhi.pkl"))

display(df.head())
display(dl_train.head())

#=============================
#     SimpleRNN Model
#=============================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences
sequence_length = 3  # Example sequence length (adjust based on your data and experimentation)
X_train, y_train = create_dataset(dl_train[features], dl_train['temperature'], sequence_length)
X_test, y_test = create_dataset(dl_test[features],dl_test['temperature'], sequence_length)


# # Build the model
# rnn_model = Sequential()
# rnn_model.add(SimpleRNN(100, activation='tanh', input_shape=(sequence_length, X_train.shape[2])))
# rnn_model.add(Dense(1))
# rnn_model.compile(optimizer='adam', loss='mse')
#
# # Define early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#
# # Train the model with early stopping
# history = rnn_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=1,
#                         callbacks=[early_stopping])
#
# # Evaluate the model
# loss = rnn_model.evaluate(X_test, y_test)
# print(f'Validation Loss: {loss}')
#
#
#
# # Make predictions
# rnn_pred = rnn_model.predict(X_test)
# rnn_pred = target_transformer.inverse_transform(rnn_pred)  # Inverse transform to original scale
#
# # Inverse transform the true values for comparison
# y_test = y_test.reshape(-1, 1)
# y_test = target_transformer.inverse_transform(y_test)
#
#
from sklearn.metrics import mean_squared_error, r2_score
#
# # Calculate RMSE and R2 scores
# rmse = np.sqrt(mean_squared_error(y_test, rnn_pred))
# r2 = r2_score(y_test, rnn_pred)
#
# print(f'RMSE: {rmse}')
# print(f'R2 Score: {r2}')
#
# # In[ ]:
#
# # Plotting the results
# plt.figure(figsize=(14, 7))
# plt.plot(df.index[-len(y_test):], y_test, label='True Values')
# plt.plot(df.index[-len(y_test):], rnn_pred, label='Predictions', linestyle='dashed')
# plt.xlabel('Date')
# plt.ylabel('Mean Temperature')
# plt.title('Mean Temperature Predictions vs True Values')
# plt.legend()
# plt.show()
#
# # In[ ]:
#
# rnn_model.summary()
#
# # In[ ]:
#
#
# # Get training and validation losses from history
# training_loss = history.history['loss']
# validation_loss = history.history['val_loss']
#
# # Plot loss values over epochs
# plt.plot(training_loss, label='Training Loss')
# plt.plot(validation_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

#=====================
# LSTM Model
#=====================
#Create sequences
sequence_length = 3  # Example sequence length (adjust based on your data and experimentation)
X_train, y_train = create_dataset(dl_train[features], dl_train['temperature'], sequence_length)
X_test, y_test = create_dataset(dl_test[features],dl_test['temperature'], sequence_length)

# ==========================================
#
#
# Unites number
#
#
# ==========================================

# from tensorflow.keras.layers import LSTM
# unit_options = [16, 32, 64, 100]
#
# results = []
#
# for units in unit_options:
#     print(f"\n\U0001F527 Training model with {units} LSTM units...")
#
#     lstm_model = Sequential()
#     lstm_model.add(LSTM(units, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
#     lstm_model.add(Dense(1))
#     lstm_model.compile(optimizer='adam', loss='mse')
#
#     # early stopping
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#
#
#     history = lstm_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=16,
#                              callbacks=[early_stopping], verbose=0)
#
#
#     loss = lstm_model.evaluate(X_test, y_test, verbose=0)
#     print(f'Validation Loss: {loss}')
#
#
#     lstm_model.summary()
#
#
#     lstm_pred = lstm_model.predict(X_test)
#     lstm_pred = target_transformer.inverse_transform(lstm_pred)
#
#     y_test_reshaped = y_test.reshape(-1, 1)
#     y_test_inv = target_transformer.inverse_transform(y_test_reshaped)
#
#
#     rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_pred))
#     r2 = r2_score(y_test_inv, lstm_pred)
#     print(f'RMSE: {rmse}')
#     print(f'R2 Score: {r2}')
#
#
#     results.append({
#         'units': units,
#         'val_loss': loss,
#         'rmse': rmse,
#         'r2': r2
#     })
#
#
#     plt.figure(figsize=(14, 7))
#     plt.plot(y_test_inv, label='True Values')
#     plt.plot(lstm_pred, label='Predictions', linestyle='dashed')
#     plt.xlabel('Time Step')
#     plt.ylabel('Mean Temperature')
#     plt.title(f'LSTM ({units} units): Predictions vs True Values')
#     plt.legend()
#     plt.show()
#
#
#     training_loss = history.history['loss']
#     validation_loss = history.history['val_loss']
#
#     plt.plot(training_loss, label='Training Loss')
#     plt.plot(validation_loss, label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss (MSE)')
#     plt.title(f'LSTM ({units} units): Training and Validation Loss')
#     plt.legend()
#     plt.show()
#
# #
# print("\n\U0001F4CA Summary of all experiments:")
# for r in results:
#     print(f"Units: {r['units']} | Val Loss: {r['val_loss']:.4f} | RMSE: {r['rmse']:.2f} | R2: {r['r2']:.4f}")
#
# unit_vals = [r['units'] for r in results]
# rmse_vals = [r['rmse'] for r in results]
# r2_vals = [r['r2'] for r in results]
#
# plt.figure(figsize=(10, 5))
# plt.plot(unit_vals, rmse_vals, marker='o', label='RMSE')
# plt.plot(unit_vals, r2_vals, marker='s', label='R² Score')
# plt.xlabel('LSTM Units')
# plt.ylabel('Score')
# plt.title('Comparison of RMSE and R² for Different LSTM Unit Sizes')
# plt.legend()
# plt.grid(True)
# plt.show()


from tensorflow.keras.layers import LSTM

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(32, activation='tanh', input_shape=(sequence_length, X_train.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = lstm_model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=16,
                         callbacks=[early_stopping])  #batch_size=1

# Evaluate the model
loss = lstm_model.evaluate(X_test, y_test)
print(f'Validation Loss: {loss}')


lstm_model.summary()

# Make predictions
lstm_pred = lstm_model.predict(X_test)
lstm_pred = target_transformer.inverse_transform(lstm_pred)  # Inverse transform to original scale

# Inverse transform the true values for comparison
y_test = y_test.reshape(-1, 1)
y_test = target_transformer.inverse_transform(y_test)
#
#
# Calculate RMSE and R2 scores
rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
r2 = r2_score(y_test, lstm_pred)

print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df.index[-len(y_test):], y_test, label='True Values')
plt.plot(df.index[-len(y_test):], lstm_pred, label='Predictions', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Mean Temperature')
plt.title('Mean Temperature Predictions vs True Values')
plt.legend()
plt.show()


# Get training and validation losses from history
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot loss values over epochs
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# To packet

model_path = os.path.join(output_dir, "lstm_temperature_model_delhi.h5")
lstm_model.save(model_path)


#====================
# Bidirectional LSTM
#====================

# #Create sequences
# sequence_length = 3  # Example sequence length (adjust based on your data and experimentation)
# X_train, y_train = create_dataset(dl_train[features], dl_train['temperature'], sequence_length)
# X_test, y_test = create_dataset(dl_test[features],dl_test['temperature'], sequence_length)

#
# from tensorflow.keras.layers import LSTM, Bidirectional
#
# # Build the bidirectional LSTM model
# model = Sequential()
# model.add(Bidirectional(LSTM(100, activation='tanh', input_shape=(sequence_length, X_train.shape[2]))))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
#
# # Define early stopping callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#
# # Train the model
# history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=1,
#                     callbacks=[early_stopping])
#
# # Evaluate the model
# loss = model.evaluate(X_test, y_test)
# print(f'Validation Loss: {loss}')
#
#
# model.summary()
#
# # Make predictions
# bilstm_pred = model.predict(X_test)
# bilstm_pred = target_transformer.inverse_transform(bilstm_pred)  # Inverse transform to original scale
#
# # Inverse transform the true values for comparison
# y_test = y_test.reshape(-1, 1)
# y_test = target_transformer.inverse_transform(y_test)
#
#
# # Calculate RMSE and R2 scores
# rmse = np.sqrt(mean_squared_error(y_test, bilstm_pred))
# r2 = r2_score(y_test, bilstm_pred)
#
# print(f'RMSE: {rmse}')
# print(f'R2 Score: {r2}')
#
# # Plot the results
# plt.figure(figsize=(14, 7))
# plt.plot(df.index[-len(y_test):], y_test, label='True Values')
# plt.plot(df.index[-len(y_test):], bilstm_pred, label='Predictions', linestyle='dashed')
# plt.xlabel('Date')
# plt.ylabel('Mean Temperature')
# plt.title('Mean Temperature Predictions vs True Values')
# plt.legend()
# plt.show()
#
#
# # Get training and validation losses from history
# training_loss = history.history['loss']
# validation_loss = history.history['val_loss']
#
# # Plot loss values over epochs
# plt.plot(training_loss, label='Training Loss')
# plt.plot(validation_loss, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()