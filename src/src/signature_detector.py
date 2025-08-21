import pandas as pd

class LiquiditySignatureDetector:
    def __init__(self, stop_hunt_threshold=0.0005):
        """
        stop_hunt_threshold: pip difference (in EUR/USD) to consider a liquidity grab
        """
        self.stop_hunt_threshold = stop_hunt_threshold

    def detect_liquidity_grabs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect liquidity grabs (stop hunts) based on wick analysis.
        Conditions:
        - If the low of a candle sweeps below the previous low but closes higher -> bullish liquidity grab.
        - If the high of a candle sweeps above the previous high but closes lower -> bearish liquidity grab.
        """

        signals = []

        for i in range(1, len(df)):
            prev_low = df.loc[i-1, "low"]
            prev_high = df.loc[i-1, "high"]

            curr_open = df.loc[i, "open"]
            curr_close = df.loc[i, "close"]
            curr_low = df.loc[i, "low"]
            curr_high = df.loc[i, "high"]

            # Bullish grab (sweep liquidity below)
            if curr_low < prev_low - self.stop_hunt_threshold and curr_close > curr_open:
                signals.append("BULL_GRAB")
            # Bearish grab (sweep liquidity above)
            elif curr_high > prev_high + self.stop_hunt_threshold and curr_close < curr_open:
                signals.append("BEAR_GRAB")
            else:
                signals.append("NONE")

        df = df.iloc[1:].copy()
        df["signal"] = signals
        return df

if __name__ == "__main__":
    # Quick test with fake data
    sample_data = {
        "open": [1.1000, 1.1010, 1.1020],
        "high": [1.1010, 1.1030, 1.1040],
        "low": [1.0990, 1.1005, 1.1010],
        "close": [1.1005, 1.1025, 1.1008],
    }
    df = pd.DataFrame(sample_data)
    detector = LiquiditySignatureDetector()
    signals_df = detector.detect_liquidity_grabs(df)
    print(signals_df)
