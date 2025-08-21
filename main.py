from src.signature_detector import LiquiditySignatureDetector
from src.risk_manager import RiskManager
from src.trade_executor import TradeExecutor

# === CONFIG ===
ACCOUNT_BALANCE = 10000   # Example account balance
RISK_PERCENTAGE = 0.01    # 1% risk per trade
SYMBOL = "EURUSD"

# Example recent price candles (O,H,L,C)
price_data = [
    (1.1040, 1.1060, 1.1030, 1.1050),
    (1.1050, 1.1070, 1.1040, 1.1065),
    (1.1065, 1.1085, 1.1055, 1.1075),
]

def main():
    # Step 1: Detect liquidity signature
    detector = LiquiditySignatureDetector(threshold=0.0010)
    signal = detector.detect(price_data)

    if signal["direction"] is None:
        print("⚠️ No liquidity signature detected. No trade taken.")
        return

    print(f"✅ Signal detected: {signal}")

    # Step 2: Risk management (1% risk)
    risk_manager = RiskManager(account_balance=ACCOUNT_BALANCE, risk_percentage=RISK_PERCENTAGE)

    entry = price_data[-1][3]  # last candle close
    stop_loss = signal["stop_loss"]
    risk_per_unit = abs(entry - stop_loss)

    lot_size = risk_manager.calculate_lot_size(risk_per_unit)

    print(f"📊 Lot size calculated: {lot_size:.2f}")

    # Step 3: Execute trade
    executor = TradeExecutor()
    trade = executor.execute_trade(
        symbol=SYMBOL,
        direction=signal["direction"],
        entry=entry,
        stop_loss=stop_loss,
        take_profit=entry + (entry - stop_loss) if signal["direction"] == "BUY" else entry - (stop_loss - entry),
        lot_size=lot_size
    )

    print(f"🚀 Trade placed: {trade}")


if __name__ == "__main__":
    main()
