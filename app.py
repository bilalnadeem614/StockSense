import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="StockSense Forecast",
    page_icon="📈",
    layout="centered"
)


# --------------------------------
# TITLE
# --------------------------------

st.title("📈 StockSense — Stock Forecast")

st.write(
    "Select a stock and click the Predict button "
    "to forecast the next stock price."
)


# --------------------------------
# STOCK DROPDOWN
# --------------------------------

ticker = st.selectbox(
    "Select a Stock",
    ["AAPL", "TSLA", "MSFT"]
)


# --------------------------------
# PREDICT BUTTON
# --------------------------------

if st.button("Predict 🚀"):

    try:

        # -------------------------
        # LOAD MODEL
        # -------------------------

        model_path = f"models/lstm_model_{ticker}.h5"

        model = load_model(model_path)


        # -------------------------
        # LOAD SCALER
        # -------------------------

        scaler_path = f"scalers/scaler_{ticker}.pkl"

        scaler = joblib.load(scaler_path)


        # -------------------------
        # DOWNLOAD STOCK DATA
        # -------------------------

        data = yf.download(
            ticker,
            period="3mo",
            progress=False
        )


        # -------------------------
        # GET LAST 60 DAYS
        # -------------------------

        close_prices = data["Close"].values

        last_60_days = close_prices[-60:]


        # -------------------------
        # RESHAPE DATA
        # -------------------------

        last_60_days = last_60_days.reshape(-1, 1)


        # -------------------------
        # SCALE DATA
        # -------------------------

        scaled_data = scaler.transform(last_60_days)


        # -------------------------
        # PREPARE FOR LSTM
        # Shape:
        # (samples, time_steps, features)
        # -------------------------

        X = np.array([scaled_data])


        # -------------------------
        # MAKE PREDICTION
        # -------------------------

        prediction = model.predict(X)


        # -------------------------
        # INVERSE SCALE
        # -------------------------

        prediction = scaler.inverse_transform(prediction)

        predicted_price = prediction[0][0]


        # -------------------------
        # DISPLAY RESULT
        # -------------------------

        st.success(
            f"Predicted Next Price for {ticker}: "
            f"${predicted_price:.2f}"
        )


        # -------------------------
        # CREATE CHART DATA
        # -------------------------

        chart_data = pd.DataFrame({
            "Actual Price": last_60_days.flatten()
        })


        # Add predicted value

        chart_data.loc[len(chart_data)] = predicted_price


        # -------------------------
        # DISPLAY CHART
        # -------------------------

        st.subheader("📊 Recent Stock Prices")

        st.line_chart(chart_data)


    except FileNotFoundError:

        st.error(
            "Model or scaler file not found. "
            "Please check your folders and filenames."
        )

    except Exception as e:

        st.error(f"An error occurred: {e}")


# --------------------------------
# DISCLAIMER
# --------------------------------

st.divider()

st.caption(
    "⚠️ Disclaimer: This application is for educational "
    "purposes only. This is not financial advice."
)