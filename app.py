# To Close Now
# Stop Streamlit
# In your terminal where Streamlit is running, press:

# mathematica
# Copy
# Edit
# Ctrl + C
# It will stop the server.

# Deactivate virtual environment
# In the same terminal:

# nginx
# Copy
# Edit
# deactivate
# This takes you out of (venv).

# Close the terminal (optional)
# Just close the window if you don’t need it.

# Next Time You Want to Run
# Open a terminal and go to your project folder:

# powershell
# Copy
# Edit
# cd "C:\Users\PRAJWAL PANDIT\Desktop\Stock_Market_Prediction_ML"
# Activate your virtual environment:

# Run this command (temporary change for this session only):

# powershell
# Copy
# Edit
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# powershell
# Copy
# Edit
# .\venv\Scripts\activate
# Run Streamlit:

# powershell
# Copy
# Edit
# streamlit run app.py


import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------
# TRAINING FUNCTION
# --------------------------
def train_and_save_model(symbol, model_file):
    st.warning(f"Training model for {symbol}... ⏳ This may take a few minutes.")

    # Download data
    start = '2012-01-01'
    end = '2022-12-21'
    data = yf.download(symbol, start, end)

    if data.empty:
        st.error(f"No data found for {symbol}. Check the symbol and try again.")
        return None

    # Prepare training data
    data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scale = scaler.fit_transform(data_train)

    x, y = [], []
    for i in range(100, data_train_scale.shape[0]):
        x.append(data_train_scale[i - 100:i])
        y.append(data_train_scale[i, 0])
    x, y = np.array(x), np.array(y)

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x, y, epochs=5, batch_size=32, verbose=1)  # reduced epochs

    # Save model
    model.save(model_file)
    st.success(f"Model for {symbol} trained and saved successfully!")
    return model

# --------------------------
# STREAMLIT APP
# --------------------------
st.header('📈  MarketVision')

# User input for stock symbol
symbol = st.text_input("Enter Stock Symbol", "ORCL").upper()
model_file = f"MODEL_{symbol}.keras"

# Load or train model
if os.path.exists(model_file):
    model = load_model(model_file)
    st.success(f"Loaded saved model for {symbol}.")
else:
    model = train_and_save_model(symbol, model_file)

if model:
    # Download full data for prediction
    start = '2012-01-01'
    end = '2022-12-21'
    data = yf.download(symbol, start, end)

    if not data.empty:
        st.subheader(f"{symbol} Stock Data")
        st.write(data)

        # Moving averages
        ma_50_days = data.Close.rolling(50).mean()
        ma_100_days = data.Close.rolling(100).mean()
        ma_200_days = data.Close.rolling(200).mean()

        # Plot moving averages
        fig1 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r', label='MA50')
        plt.plot(data.Close, 'g', label='Close Price')
        plt.legend()
        st.pyplot(fig1)

        fig2 = plt.figure(figsize=(8, 6))
        plt.plot(ma_50_days, 'r', label='MA50')
        plt.plot(ma_100_days, 'b', label='MA100')
        plt.plot(data.Close, 'g', label='Close Price')
        plt.legend()
        st.pyplot(fig2)

        fig3 = plt.figure(figsize=(8, 6))
        plt.plot(ma_100_days, 'r', label='MA100')
        plt.plot(ma_200_days, 'b', label='MA200')
        plt.plot(data.Close, 'g', label='Close Price')
        plt.legend()
        st.pyplot(fig3)

        # Prepare test data for prediction
        data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

        scaler = MinMaxScaler(feature_range=(0, 1))
        pas_100_days = data_train.tail(100)
        data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
        data_test_scale = scaler.fit_transform(data_test)

        x_test, y_test = [], []
        for i in range(100, data_test_scale.shape[0]):
            x_test.append(data_test_scale[i - 100:i])
            y_test.append(data_test_scale[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)

        # Predict
        y_pred = model.predict(x_test)

        scale = 1 / scaler.scale_
        y_pred = y_pred * scale
        y_test = y_test * scale

        # Plot predictions
        st.subheader('Original Price vs Predicted Price')
        fig4 = plt.figure(figsize=(10, 8))
        plt.plot(y_pred, 'r', label='Predicted Price')
        plt.plot(y_test, 'g', label='Original Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig4)

