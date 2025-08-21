class RiskManager:
    def __init__(self, account_balance: float, risk_per_trade: float = 0.01):
        """
        account_balance: current trading account balance (e.g. 10,000 USD)
        risk_per_trade: fraction of balance to risk (default 1%)
        """
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(self, entry: float, stop_loss: float, pip_value: float = 10.0) -> float:
        """
        Calculate lot size based on account balance and stop loss distance.
        entry: trade entry price
        stop_loss: stop loss price
        pip_value: value per pip per lot (default $10 for EUR/USD 1 lot)
        """
        # Risk in $ for this trade
        risk_amount = self.account_balance * self.risk_per_trade

        # Distance to stop loss in pips
        stop_distance_pips = abs(entry - stop_loss) * 10000  # EUR/USD pip = 0.0001

        if stop_distance_pips == 0:
            return 0  # Avoid division by zero

        # Lot size formula
        lot_size = risk_amount / (stop_distance_pips * pip_value)
        return round(lot_size, 2)


if __name__ == "__main__":
    # Example test
    rm = RiskManager(account_balance=10000, risk_per_trade=0.01)
    entry_price = 1.1050
    stop_loss_price = 1.1020
    lot_size = rm.calculate_position_size(entry=entry_price, stop_loss=stop_loss_price)
    print(f"Recommended lot size: {lot_size} lots")
