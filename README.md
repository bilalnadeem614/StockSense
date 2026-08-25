# StockSense

A deep learning web app that forecasts stock price trends using an LSTM (Long Short-Term Memory) neural network trained on historical price data.

**Live App:** _(add deployed link here once live)_

## What It Does

1. Pick a stock ticker (e.g., `AAPL`, `TSLA`, `MSFT`).
2. The app fetches recent historical price data automatically.
3. The trained LSTM model looks at the last 60 days of prices and predicts the next price movement.
4. A chart displays the actual vs. predicted price trend.

## How It Works (High Level)

- Historical daily closing prices are pulled for a chosen stock.
- The price history is split into overlapping 60-day windows, where the model learns to predict the next day's price from the days before it.
- Prices are scaled into a smaller numeric range to help the model train effectively.
- An LSTM neural network is trained on these windows — unlike traditional models, it accounts for the *order* and *trend* of the data over time.
- The trained model is tested on unseen price data to check how closely its predictions track reality.

## Tech Stack

- **Python** — model training
- **yfinance** — historical stock data (no manual dataset needed)
- **TensorFlow / Keras** — LSTM model
- **Streamlit** — GUI and deployment

## Project Files

- `app.py` — Streamlit application (GUI)
- `train_lstm.py` — fetches stock data and trains the LSTM model
- `lstm_model.h5` — saved trained model
- `requirements.txt` — dependencies

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

This project is built for educational purposes to demonstrate time-series forecasting with LSTM. Stock prices are influenced by many real-world factors (news, sentiment, market events) beyond historical price patterns, so predictions here should **not** be used for actual financial or investment decisions.
