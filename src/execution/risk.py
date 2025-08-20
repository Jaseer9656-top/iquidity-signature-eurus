class RiskManager:
    """
    EURUSD pip value ≈ $10 per standard lot.
    Ensures risk_per_trade (e.g., 0.01 = 1%) max loss at SL.
    """
    def __init__(self, balance: float, risk_per_trade: float = 0.01, pip_value: float = 10.0):
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.pip_value = pip_value

    def compute_lots(self, stop_loss_pips: float) -> float:
        if stop_loss_pips <= 0:
            return 0.0
        risk_amount = self.balance * self.risk_per_trade
        lots = risk_amount / (stop_loss_pips * self.pip_value)
        return round(max(lots, 0.0), 2)  # broker min step usually 0.01 lot
