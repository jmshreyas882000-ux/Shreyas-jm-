import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.title("📈 Advanced AI Financial Analyst")

# Input
stock = st.text_input("Enter Stock Symbol", "AAPL")

# Load Data
data = yf.download(stock, start="2020-01-01")

st.subheader("Raw Data")
st.write(data.tail())

# Moving Averages
data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()

# Prepare Data
data["Prediction"] = data["Close"].shift(-30)

X = np.array(data["Close"]).reshape(-1, 1)
X = X[:-30]

y = np.array(data["Prediction"])
y = y[:-30]

# Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Models
lr = LinearRegression()
rf = RandomForestRegressor()

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Metrics
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

st.subheader("Model Performance")
st.write(f"Linear Regression RMSE: {lr_rmse:.2f}")
st.write(f"Random Forest RMSE: {rf_rmse:.2f}")

# Future Prediction
future = np.array(data["Close"].tail(30)).reshape(-1, 1)
future_pred = rf.predict(future)

# Buy/Sell Signal
if data["MA50"].iloc[-1] > data["MA200"].iloc[-1]:
    signal = "BUY 📈"
else:
    signal = "SELL 📉"

st.subheader("Trading Signal")
st.success(signal)

# Plot
st.subheader("Stock Chart")

plt.figure(figsize=(10,5))
plt.plot(data["Close"], label="Price")
plt.plot(data["MA50"], label="MA50")
plt.plot(data["MA200"], label="MA200")

plt.plot(range(len(data)-len(future_pred), len(data)), future_pred, label="Prediction", color="red")

plt.legend()
st.pyplot(plt)

st.success("Analysis Complete 🚀")
