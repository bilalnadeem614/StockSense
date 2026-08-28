!pip install yfinance tensorflow scikit-learn -q

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle

# ---- Settings ----
TICKERS = ["AAPL", "TSLA", "MSFT"]   
WINDOW_SIZE = 60                      # how many past days used to predict next day

# ---- Loop through each stock ----
for TICKER in TICKERS:
    print(f"\n===== Training model for {TICKER} =====")

    # Step 1: Get historical data
    data = yf.download(TICKER, period="5y")
    prices = data[['Close']].values

    # Step 2: Scale prices between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Step 3: Create windowed sequences
    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_prices)):
        X.append(scaled_prices[i-WINDOW_SIZE:i, 0])
        y.append(scaled_prices[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Step 4: Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Step 5: Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 6: Train
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    # Step 7: Save model + scaler with ticker name in filename
    model.save(f"lstm_model_{TICKER}.h5")
    with open(f"scaler_{TICKER}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved: lstm_model_{TICKER}.h5 and scaler_{TICKER}.pkl")

print("\nAll 3 models trained and saved.")
