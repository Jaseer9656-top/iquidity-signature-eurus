import pandas as pd
import ccxt
import datetime

class DataLoader:
    def __init__(self, exchange="oanda", symbol="EUR/USD", timeframe="1h", limit=1000):
        self.symbol = symbol.replace("/", "")
        self.timeframe = timeframe
        self.limit = limit

        # Use OANDA or any supported CCXT exchange
        if exchange == "oanda":
            self.exchange = ccxt.oanda({
                "enableRateLimit": True,
            })
        else:
            raise ValueError("Only OANDA is supported for now.")

    def fetch_data(self):
        """Fetch historical OHLCV data"""
        ohlcv = self.exchange.fetch_ohlcv(
            "EUR/USD",
            timeframe=self.timeframe,
            limit=self.limit
        )

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

if __name__ == "__main__":
    loader = DataLoader(timeframe="1h", limit=500)
    df = loader.fetch_data()
    print(df.head())
